#!/usr/bin/env python3
"""
geometry_tracker.py

Detects cylindrical objects from a depth image using vertical line pairs.
A detection is 'confirmed' the moment a valid left-right line pair is found
(i.e. the existing _pair_lines_into_cylinders logic returns a result).

The center between the two lines is the cylinder's pixel-space position.
A confirmed detection is published on /geometry/confirmed_cylinder as a
Float32MultiArray:

    [cx_px, cy_px, depth_m, width_px]

Confirmation is GATED: messages are only published while /mission/search_active
carries True.  This prevents takeoff detections from being stored.

Deduplication: once a cylinder is confirmed, further confirmations are
suppressed for CONFIRM_COOLDOWN_S seconds.  This prevents the same physical
cylinder being stored multiple times during a slow yaw scan.

During approach the live detection is always published on /geometry/cylinder_center
(Point: x=cx_px, y=cy_px, z=depth_m) so the executor can read it to center
the object in frame before advancing.
"""

import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from std_msgs.msg import Bool, Float32MultiArray
from geometry_msgs.msg import Point


class GeometryTracker(Node):

    # -----------------------------------------------------------------------
    # Tuning constants
    # -----------------------------------------------------------------------

    # Seconds of silence after each confirmed detection.
    # During a ~0.5 rad/s yaw scan a cylinder stays in frame for ~4-6 s,
    # so 3 s cooldown means it can only be stored once per pass.
    CONFIRM_COOLDOWN_S: float = 3.0

    def __init__(self):
        super().__init__('geometry_tracker')

        self.cv_bridge = CvBridge()

        # ---- Gate flag: only confirm during search ----
        self._search_active: bool = False

        # ---- Cooldown: wall-clock time of last confirmed publication ----
        self._last_confirm_time: float = 0.0

        # Subscribers
        self.create_subscription(
            Image, '/drone/front_depth',
            self.depth_image_callback, 10)
        self.create_subscription(
            Bool, '/mission/search_active',
            self._search_active_callback, 10)

        # Publishers
        self.debug_image_pub   = self.create_publisher(Image,             '/geometry/debug_image',        10)
        self.cylinder_pose_pub = self.create_publisher(Point,             '/geometry/cylinder_center',    10)  # live detection for approach centering
        self.cylinder_info_pub = self.create_publisher(Float32MultiArray, '/geometry/cylinder_info',      10)  # kept for backwards compat
        self.confirmed_pub     = self.create_publisher(Float32MultiArray, '/geometry/confirmed_cylinder', 10)  # gated + cooldown-deduped

        self.get_logger().info('Geometry tracker node initialised (line-pair confirmation mode)')

    # -----------------------------------------------------------------------
    # Search-active gate callback
    # -----------------------------------------------------------------------

    def _search_active_callback(self, msg: Bool):
        was = self._search_active
        self._search_active = msg.data
        if msg.data and not was:
            self._last_confirm_time = 0.0   # reset cooldown at search start
            self.get_logger().info('[TRACKER] Search gate OPEN ? confirmations enabled')
        elif not msg.data and was:
            self.get_logger().info('[TRACKER] Search gate CLOSED ? confirmations suppressed')

    # -----------------------------------------------------------------------
    # Step 1 ? Depth preprocessing
    # -----------------------------------------------------------------------

    def _preprocess_depth(self, depth_image: np.ndarray) -> np.ndarray:
        """
        Convert raw 32FC1 depth to a uint8 grey image suitable for edge
        detection.  CLAHE gives stable local contrast without the
        frame-to-frame flicker of global histogram equalisation.
        """
        depth_min, depth_max = 0.0, 10.0

        disp = depth_image.copy()
        disp[np.isnan(disp)] = 0.0
        disp = np.clip(disp, depth_min, depth_max)
        disp = ((disp - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(disp)

    # -----------------------------------------------------------------------
    # Step 2 ? Vertical line detection
    # -----------------------------------------------------------------------

    def _find_vertical_lines(self, grey: np.ndarray):
        """
        Canny + HoughLinesP, filtered to near-vertical lines only.
        Returns (edges, vertical_lines) where vertical_lines is a list of
        [x1, y1, x2, y2] arrays.
        """
        blurred = cv2.GaussianBlur(grey, (5, 5), 0)
        edges   = cv2.Canny(blurred, 50, 150)
        lines   = cv2.HoughLinesP(edges, 1, np.pi / 180, 50,
                                   minLineLength=80, maxLineGap=15)
        vertical = []
        if lines is not None:
            for ln in lines:
                x1, y1, x2, y2 = ln[0]
                angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi)
                if 70.0 < angle < 110.0:
                    vertical.append([x1, y1, x2, y2])
        return edges, vertical

    # -----------------------------------------------------------------------
    # Step 3 ? Cluster vertical lines by x-position
    # -----------------------------------------------------------------------

    @staticmethod
    def _cluster_lines(vertical_lines: list, gap_threshold: float = 15.0) -> list:
        """
        Group lines whose mean-x values are within `gap_threshold` pixels of
        each other into clusters and return one representative per cluster
        (the one closest to the cluster centroid).

        Prevents the same physical edge from generating two slightly different
        lines each frame, which destabilises pairing.
        """
        if not vertical_lines:
            return []

        descriptors = []
        for x1, y1, x2, y2 in vertical_lines:
            descriptors.append({
                'mean_x': (x1 + x2) / 2.0,
                'y_min':  min(y1, y2),
                'y_max':  max(y1, y2),
                'raw':    [x1, y1, x2, y2],
            })

        descriptors.sort(key=lambda d: d['mean_x'])

        clusters: list[list] = [[descriptors[0]]]
        for d in descriptors[1:]:
            if d['mean_x'] - clusters[-1][-1]['mean_x'] <= gap_threshold:
                clusters[-1].append(d)
            else:
                clusters.append([d])

        representatives = []
        for cluster in clusters:
            centroid_x = sum(d['mean_x'] for d in cluster) / len(cluster)
            best = min(cluster, key=lambda d: abs(d['mean_x'] - centroid_x))
            representatives.append(best)

        return representatives

    # -----------------------------------------------------------------------
    # Step 4 ? Pair clustered lines into cylinder hypotheses
    # -----------------------------------------------------------------------

    @staticmethod
    def _pair_lines_into_cylinders(representatives: list,
                                    min_gap: float = 20.0,
                                    max_gap: float = 200.0) -> list:
        """
        Greedy left-right pairing of clustered line representatives.

        Returns a list of dicts:
            { 'cx', 'cy', 'width_px', 'height_px',
              'left_rep', 'right_rep' }   <-- reps kept for debug drawing
        """
        cylinders = []
        used = set()

        for i in range(len(representatives)):
            if i in used:
                continue
            for j in range(i + 1, len(representatives)):
                if j in used:
                    continue
                gap = representatives[j]['mean_x'] - representatives[i]['mean_x']
                if gap < min_gap:
                    continue
                if gap > max_gap:
                    break

                li, lj    = representatives[i], representatives[j]
                cx        = (li['mean_x'] + lj['mean_x']) / 2.0   # midpoint between lines
                y_top     = min(li['y_min'], lj['y_min'])
                y_bot     = max(li['y_max'], lj['y_max'])
                cy        = (y_top + y_bot) / 2.0
                width_px  = gap
                height_px = float(y_bot - y_top)

                cylinders.append({
                    'cx': cx, 'cy': cy,
                    'width_px': width_px, 'height_px': height_px,
                    'left_rep': li, 'right_rep': lj,
                })
                used.add(i)
                used.add(j)
                break

        return cylinders

    # -----------------------------------------------------------------------
    # Cooldown deduplication helper
    # -----------------------------------------------------------------------

    def _cooldown_elapsed(self) -> bool:
        """True when enough time has passed since the last confirmed publication."""
        return (time.monotonic() - self._last_confirm_time) >= self.CONFIRM_COOLDOWN_S

    # -----------------------------------------------------------------------
    # Main callback
    # -----------------------------------------------------------------------

    def depth_image_callback(self, msg: Image):
        try:
            depth_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            img_h, img_w = depth_image.shape[:2]

            # ---- Preprocessing ----
            grey        = self._preprocess_depth(depth_image)
            debug_image = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)

            # ---- Vertical line detection ----
            edges, vertical_lines = self._find_vertical_lines(grey)

            # Draw raw vertical lines in dim red
            for x1, y1, x2, y2 in vertical_lines:
                cv2.line(debug_image, (x1, y1), (x2, y2), (0, 0, 180), 1)

            # ---- Cluster lines ----
            reps = self._cluster_lines(vertical_lines, gap_threshold=15.0)

            # Draw cluster representatives in bright red
            for rep in reps:
                x1, y1, x2, y2 = rep['raw']
                cv2.line(debug_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # ---- Pair into cylinder hypotheses ----
            cylinders = self._pair_lines_into_cylinders(reps)

            for cyl in cylinders:
                cx_px = int(np.clip(cyl['cx'],  0, img_w - 1))
                cy_px = int(np.clip(cyl['cy'],  0, img_h - 1))

                depth_m = float(depth_image[cy_px, cx_px])
                if np.isnan(depth_m) or depth_m <= 0.0:
                    continue

                width_px  = cyl['width_px']
                height_px = cyl['height_px']

                # ---- Always publish live cylinder center (used by approach centering) ----
                center_msg   = Point()
                center_msg.x = float(cx_px)
                center_msg.y = float(cy_px)
                center_msg.z = depth_m
                self.cylinder_pose_pub.publish(center_msg)

                info_msg      = Float32MultiArray()
                info_msg.data = [float(width_px), float(height_px), depth_m, 1.0, 1.0]
                self.cylinder_info_pub.publish(info_msg)

                # ---- Confirmed publication: GATED (search only) + COOLDOWN dedup ----
                # The green box always draws so the operator can see detections,
                # but /geometry/confirmed_cylinder only fires when both conditions met.
                if self._search_active and self._cooldown_elapsed():
                    confirmed_msg      = Float32MultiArray()
                    confirmed_msg.data = [
                        float(cx_px),    # pixel center x
                        float(cy_px),    # pixel center y
                        depth_m,         # depth in metres at that pixel
                        float(width_px), # apparent width in pixels
                    ]
                    self.confirmed_pub.publish(confirmed_msg)
                    self._last_confirm_time = time.monotonic()

                    self.get_logger().info(
                        f'[CONFIRMED] Cylinder at px=({cx_px},{cy_px}) '
                        f'depth={depth_m:.2f}m width={width_px:.0f}px'
                    )

                # ---- Draw detection box ----
                # Green when confirmed (search active + cooldown ok), orange otherwise
                in_cooldown = not self._cooldown_elapsed()
                box_color = (0, 220, 0) if self._search_active and not in_cooldown else (0, 140, 255)

                hw = int(width_px  / 2)
                hh = int(height_px / 2)
                cv2.rectangle(debug_image,
                              (cx_px - hw, cy_px - hh),
                              (cx_px + hw, cy_px + hh),
                              box_color, 2)
                cv2.line(debug_image, (cx_px - 18, cy_px), (cx_px + 18, cy_px), box_color, 2)
                cv2.line(debug_image, (cx_px, cy_px - 18), (cx_px, cy_px + 18), box_color, 2)

                cooldown_remaining = max(0.0, self.CONFIRM_COOLDOWN_S - (time.monotonic() - self._last_confirm_time))
                label = f'D:{depth_m:.2f}m'
                if in_cooldown:
                    label += f' cd:{cooldown_remaining:.1f}s'
                cv2.putText(debug_image, label,
                            (cx_px - 55, cy_px - 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1)

            # ---- Image-center crosshair ----
            cx_c, cy_c = img_w // 2, img_h // 2
            cv2.line(debug_image, (cx_c - 20, cy_c), (cx_c + 20, cy_c), (0, 255, 255), 2)
            cv2.line(debug_image, (cx_c, cy_c - 20), (cx_c, cy_c + 20), (0, 255, 255), 2)
            center_depth = depth_image[cy_c, cx_c]
            if not np.isnan(center_depth):
                cv2.putText(debug_image,
                            f'Center: {center_depth:.2f}m',
                            (10, img_h - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

            # ---- Legend + gate status ----
            gate_str  = 'OPEN' if self._search_active else 'CLOSED'
            gate_color = (0, 220, 0) if self._search_active else (0, 0, 220)
            cv2.putText(debug_image, 'Dim red:    Raw vertical lines',       (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,   0, 180), 1)
            cv2.putText(debug_image, 'Bright red: Cluster reps',             (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,   0, 255), 1)
            cv2.putText(debug_image, 'Green:      Confirmed (gated)',        (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220,   0), 1)
            cv2.putText(debug_image, 'Orange:     Detected (not confirmed)', (10, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 255), 1)
            cv2.putText(debug_image, f'Search gate: {gate_str}',             (10, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.45, gate_color,    1)

            # ---- Publish debug image ----
            debug_msg        = self.cv_bridge.cv2_to_imgmsg(debug_image, encoding='bgr8')
            debug_msg.header = msg.header
            self.debug_image_pub.publish(debug_msg)

        except Exception as e:
            self.get_logger().error(f'Error processing depth image: {str(e)}')


def main():
    rclpy.init()
    node = GeometryTracker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
