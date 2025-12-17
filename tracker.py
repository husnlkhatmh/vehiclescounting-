import math

class Tracker:
    def __init__(self):
        self.center_points = {}
        self.id_count = 0

    def update(self, objects_rect):
        objects_bbs_ids = []

        for rect in objects_rect:
            x1, y1, x2, y2 = rect
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            same_object = False
            for oid, center in self.center_points.items():
                dist = math.hypot(cx - center[0], cy - center[1])
                if dist < 35:
                    self.center_points[oid] = (cx, cy)
                    objects_bbs_ids.append([x1, y1, x2, y2, oid])
                    same_object = True
                    break

            if not same_object:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x1, y1, x2, y2, self.id_count])
                self.id_count += 1

        new_centers = {}
        for obj in objects_bbs_ids:
            _, _, _, _, oid = obj
            new_centers[oid] = self.center_points[oid]

        self.center_points = new_centers.copy()
        return objects_bbs_ids