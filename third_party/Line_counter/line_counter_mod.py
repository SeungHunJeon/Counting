from typing import Dict

import cv2
import numpy as np

from supervision.draw.color import Color
from supervision.geometry.dataclasses import Point, Rect, Vector
from supervision.tools.detections import Detections


class LineCounter:
    def __init__(self, start: None, end: None, class_id: None, class_name_dict: None):
        """
        Initialize a LineCounter object.

        :param start: Point : The starting point of the line.
        :param end: Point : The ending point of the line.
        """
        self.vector_batch = []
        self.start_points = start
        self.end_points = end
        for i in range(len(self.start_points)):
            self.vector_batch.append(Vector(start=self.start_points[i], end=self.end_points[i]))
        self.tracker_state: Dict[str, bool] = {}
        self.tracker_line: Dict[str, int] = {}
        self.in_count_dict: Dict[str, int] = {}
        self.out_count_dict: Dict[str, int] = {}
        self.in_count_dict_batch = []
        self.out_count_dict_batch = []
        self.tracker_state_batch = []
        # self.tracker_class_id_dict: Dict[str, int] = {}
        self.in_count: int = 0
        self.out_count: int = 0
        self.class_id = class_id
        self.class_name_dict = class_name_dict
        for id in self.class_id:
            self.in_count_dict[self.class_name_dict[id]] = 0
            self.out_count_dict[self.class_name_dict[id]] = 0
        
        for i in range(len(self.start_points)):
            self.in_count_dict_batch.append(self.in_count_dict.copy())
            self.out_count_dict_batch.append(self.out_count_dict.copy())
            self.tracker_state_batch.append(self.tracker_state.copy())
            

    def update(self, detections: Detections):
        """
        Update the in_count and out_count for the detections that cross the line.

        :param detections: Detections : The detections for which to update the counts.
        """
        for xyxy, confidence, class_id, tracker_id in detections:
            
            # handle detections with no tracker_id
            if tracker_id is None:
                continue

            # we check if all four anchors of bbox are on the same side of vector
            x1, y1, x2, y2 = xyxy
            anchors = [
                Point(x=x1, y=y1),
                Point(x=x1, y=y2),
                Point(x=x2, y=y1),
                Point(x=x2, y=y2),
            ]
            mean_anchor = Point(x=(x1+x2)/2, y=(y1+y2)/2)
            triggers_batch = []
            for i in range(len(self.vector_batch)):
                # triggers = [self.vector_batch[i].is_in(point=anchor) for anchor in anchors]
                triggers = self.vector_batch[i].is_in(point=mean_anchor)
                triggers_batch.append(triggers)

            for id, triggers in enumerate(triggers_batch):
                # detection is partially in and partially out
                # if len(set(triggers)) == 2:
                #     continue

                # tracker_state = triggers[0]
                tracker_state = triggers
                # handle new detection, if not in the batch, append it
                if tracker_id not in self.tracker_state_batch[id]:
                    self.tracker_state_batch[id][tracker_id] = tracker_state
                    self.tracker_line[tracker_id] = [class_id]
                    continue

                # handle detection on the same side of the line
                if self.tracker_state_batch[id].get(tracker_id) == tracker_state:
                    continue

                # If the detection box cross-over the line
                self.tracker_state_batch[id][tracker_id] = tracker_state
                # print(self.tracker_state_batch)

                self.tracker_line[tracker_id].append(id)
                if tracker_state:
                    self.in_count_dict_batch[id][self.class_name_dict[class_id]] += 1
                else:
                    self.out_count_dict_batch[id][self.class_name_dict[class_id]] += 1

            


class LineCounterAnnotator:
    def __init__(
        self,
        thickness: float = 2,
        color: Color = Color.white(),
        text_thickness: float = 2,
        text_color: Color = Color.black(),
        text_scale: float = 0.5,
        text_offset: float = 1.5,
        text_padding: int = 10
    ):
        """
        Initialize the LineCounterAnnotator object with default values.

        :param thickness: float : The thickness of the line that will be drawn.
        :param color: Color : The color of the line that will be drawn.
        :param text_thickness: float : The thickness of the text that will be drawn.
        :param text_color: Color : The color of the text that will be drawn.
        :param text_scale: float : The scale of the text that will be drawn.
        :param text_offset: float : The offset of the text that will be drawn.
        :param text_padding: int : The padding of the text that will be drawn.
        """
        self.thickness: float = thickness
        self.color: Color = color
        self.text_thickness: float = text_thickness
        self.text_color: Color = text_color
        self.text_scale: float = text_scale
        self.text_offset: float = text_offset
        self.text_padding: int = text_padding
        self.color_batch = [Color.white(), Color.red(), Color.green(), Color.blue()]

    def result(self, line_counter: LineCounter):
        result_batch = []
        for i in range(len(line_counter.vector_batch)):
            in_dict = dict()
            out_dict = dict()
            for id in line_counter.in_count_dict_batch[i]:
                in_dict[id] = line_counter.in_count_dict_batch[i][id]
                out_dict[id] = line_counter.out_count_dict_batch[i][id]
            result_batch.append([in_dict, out_dict])
        return result_batch

    def annotate(self, frame: np.ndarray, line_counter: LineCounter) -> np.ndarray:
        """
        Draws the line on the frame using the line_counter provided.

        :param frame: np.ndarray : The image on which the line will be drawn
        :param line_counter: LineCounter : The line counter that will be used to draw the line
        :return: np.ndarray : The image with the line drawn on it
        """

        for i in range(len(line_counter.vector_batch)):
            cv2.line(
                frame,
                line_counter.vector_batch[i].start.as_xy_int_tuple(),
                line_counter.vector_batch[i].end.as_xy_int_tuple(),
                self.color_batch[i].as_bgr(),
                self.thickness,
                lineType=cv2.LINE_AA,
                shift=0,
            )
            cv2.circle(
                frame,
                line_counter.vector_batch[i].start.as_xy_int_tuple(),
                radius=5,
                color=self.text_color.as_bgr(),
                thickness=-1,
                lineType=cv2.LINE_AA,
            )
            cv2.circle(
                frame,
                line_counter.vector_batch[i].end.as_xy_int_tuple(),
                radius=5,
                color=self.text_color.as_bgr(),
                thickness=-1,
                lineType=cv2.LINE_AA,
            )
        
            in_text_result = f"# {i}, (in) "
            out_text_result = f"# {i}, (out) "

            for id in line_counter.in_count_dict_batch[i]:
                in_text = f"{id} : {line_counter.in_count_dict_batch[i][id]} "
                in_text_result += in_text

            for id in line_counter.out_count_dict_batch[i]:
                out_text = f"{id} : {line_counter.out_count_dict_batch[i][id]} "
                out_text_result += out_text

            # in_text = f"in: {line_counter.in_count}"
            # out_text = f"out: {line_counter.out_count}"

            in_text = in_text_result
            out_text = out_text_result

            (in_text_width, in_text_height), _ = cv2.getTextSize(
                in_text, cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, self.text_thickness
            )
            (out_text_width, out_text_height), _ = cv2.getTextSize(
                out_text, cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, self.text_thickness
            )

            in_text_x = int(
                (line_counter.vector_batch[i].end.x + line_counter.vector_batch[i].start.x - in_text_width)
                / 2
            )
            in_text_y = int(
                (line_counter.vector_batch[i].end.y + line_counter.vector_batch[i].start.y + in_text_height)
                / 2
                - self.text_offset * in_text_height
            )

            out_text_x = int(
                (line_counter.vector_batch[i].end.x + line_counter.vector_batch[i].start.x - out_text_width)
                / 2
            )
            out_text_y = int(
                (line_counter.vector_batch[i].end.y + line_counter.vector_batch[i].start.y + out_text_height)
                / 2
                + self.text_offset * out_text_height
            )

            in_text_background_rect = Rect(
                x=in_text_x,
                y=in_text_y - in_text_height,
                width=in_text_width,
                height=in_text_height,
            ).pad(padding=self.text_padding)
            out_text_background_rect = Rect(
                x=out_text_x,
                y=out_text_y - out_text_height,
                width=out_text_width,
                height=out_text_height,
            ).pad(padding=self.text_padding)

            cv2.rectangle(
                frame,
                in_text_background_rect.top_left.as_xy_int_tuple(),
                in_text_background_rect.bottom_right.as_xy_int_tuple(),
                self.color_batch[i].as_bgr(),
                -1,
            )
            cv2.rectangle(
                frame,
                out_text_background_rect.top_left.as_xy_int_tuple(),
                out_text_background_rect.bottom_right.as_xy_int_tuple(),
                self.color_batch[i].as_bgr(),
                -1,
            )

            cv2.putText(
                frame,
                in_text,
                (in_text_x, in_text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.text_scale,
                self.text_color.as_bgr(),
                self.text_thickness,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                out_text,
                (out_text_x, out_text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.text_scale,
                self.text_color.as_bgr(),
                self.text_thickness,
                cv2.LINE_AA,
            )
