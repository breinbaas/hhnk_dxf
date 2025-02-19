import ezdxf
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as mplPolygon
from shapely import Polygon, LineString
from shapely.ops import polygonize, unary_union, orient
import numpy as np
import math
from leveelogic.objects.levee import Levee
from leveelogic.objects.soilpolygon import SoilPolygon
from leveelogic.objects.soil import Soil
from leveelogic.helpers import case_insensitive_glob
from pathlib import Path


DXF_PATH = "data"
EPSILON = 1e-3
DEBUG = True


#####################
# AI generated code #
#####################
def generate_random_color():
    """Generates a random RGB color as a tuple."""
    return tuple(np.random.rand(3))


#####################
# AI generated code #
#####################
def find_polygons(lines, remove_overlapping=False):
    """
    Finds polygons from a list of line segments.

    Args:
        lines: A list of line segments, where each segment is a tuple ((x1, y1), (x2, y2)).

    Returns:
        A list of polygons, where each polygon is a list of points.
    """

    graph = {}
    for p1, p2 in lines:
        if p1 not in graph:
            graph[p1] = []
        if p2 not in graph:
            graph[p2] = []
        graph[p1].append(p2)
        graph[p2].append(p1)  # If undirected

    polygons = []
    visited = set()

    def find_cycles(start_point, current_point, path):
        visited.add(current_point)
        path.append(current_point)

        for neighbor in graph.get(current_point, []):
            if neighbor == start_point and len(path) > 2:  # Found a cycle
                polygons.append(path[:])  # Append a copy of the path
            elif neighbor not in visited:
                find_cycles(start_point, neighbor, path[:])
        visited.remove(
            current_point
        )  # remove the current point from the visited set to allow other paths to use it.

    for start_point in graph:
        find_cycles(start_point, start_point, [])

    # Remove duplicates (cycles found in both directions)
    unique_polygons = []
    seen = set()
    for polygon in polygons:
        sorted_polygon = tuple(sorted(polygon))
        if sorted_polygon not in seen:
            seen.add(sorted_polygon)
            unique_polygons.append(Polygon(polygon))

    # #########################
    # # end AI generated code #
    # #########################
    if remove_overlapping:
        # Remove polygons that overlap other polygons
        final_polygons = []
        for i in range(len(unique_polygons)):
            is_overlapping = False
            for j in range(len(unique_polygons)):
                if i != j:
                    if unique_polygons[i].overlaps(
                        unique_polygons[j]
                    ) or unique_polygons[j].within(unique_polygons[i]):
                        is_overlapping = True
                        break

            if not is_overlapping:
                final_polygons.append(unique_polygons[i])
    else:
        final_polygons = unique_polygons

    return final_polygons


class DXFModel:
    def __init__(self):
        self.lines = []
        self.polygons = []

    @property
    def xmin(self):
        return min([p[0] for p in self.points])

    @property
    def xmax(self):
        return max([p[0] for p in self.points])

    @property
    def points(self):
        points = []
        for line in self.lines:
            points += [line[0], line[1]]

        return list(set(points))

    @property
    def shared_points(self):
        result = []
        for p in self.points:
            inum = 0
            for l in self.lines:
                if l[0] == p or l[1] == p:
                    inum += 1
            result.append((inum, p))

        return [p[1] for p in result if p[0] > 2]

    @classmethod
    def from_dxf(cls, filename: str) -> "DXFModel":
        result = DXFModel()

        doc = ezdxf.readfile(filename)
        msp = doc.modelspace()

        lines = []
        points = []

        # lees de polylijnen in
        print("Lijnen aan het inlezen...")
        for polyline in msp.query("LWPOLYLINE"):
            points = [
                (round(float(p[0]), 2), round(float(p[1]), 2))
                for p in polyline.vertices()
            ]

            for i in range(1, len(points)):
                p1 = points[i - 1]
                p2 = points[i]
                lines.append(((p1[0], p1[1]), (p2[0], p2[1])))
                points += [p1, p2]

        # lees de losse lijnen in
        for line in msp.query("LINE"):
            p1 = (round(line.dxf.start[0], 2), round(line.dxf.start[1], 2))
            p2 = (round(line.dxf.end[0], 2), round(line.dxf.end[1], 2))
            lines.append((p1, p2))
            points += [p1, p2]

        # lees polylijnen in
        for polyline in msp.query("POLYLINE"):
            polyline_points = [
                (round(p[0], 2), round(p[1], 2)) for p in polyline.points()
            ]
            points += polyline_points
            for i in range(1, len(polyline_points)):
                p1 = polyline_points[i - 1]
                p2 = polyline_points[i]
                lines.append((p1, p2))

        # remove duplicates
        points = list(set(points))

        #####################################################################################
        # de linker en rechter lijn zijn nu doorgaande lijnen die van boven tot onder lopen #
        # deze moeten vervangen worden door individuele lijnen waarbij de snijpunten        #
        # met de laagscheidingen worden toegevoegd                                          #
        #####################################################################################
        print("Toevoegen laagscheidingspunten aan de linker- en rechterzijde...")

        # vind de index van deze lijnen
        idx_left, idx_right = -1, -1

        xmin = min([p[0] for p in points])
        xmax = max([p[0] for p in points])

        for i, l in enumerate(lines):
            if l[0][0] == l[1][0] and l[0][0] == xmin:
                idx_left = i
            if l[0][0] == l[1][0] and l[0][0] == xmax:
                idx_right = i

        line_left = lines[idx_left]
        line_right = lines[idx_right]

        ytop_left = max([line_left[0][1], line_left[1][1]])
        ybottom_left = min([line_left[0][1], line_left[1][1]])
        ytop_right = max([line_right[0][1], line_right[1][1]])
        ybottom_right = min([line_right[0][1], line_right[1][1]])

        ############################################################################
        # vervang deze lijnen door lijnen die alle tussenliggende punten ook raken #
        ############################################################################

        # zoek naar alle lijnen met xmin of xmax als startpunt
        yt_left = max([l])
        lines_left = [l for l in lines if l[0][0] == xmin or l[1][0] == xmin]
        lines_right = [l for l in lines if l[0][0] == xmax or l[1][0] == xmax]

        ys_left, ys_right = [], []
        for p1, p2 in lines_left:
            if p1[0] == xmin:
                ys_left.append(p1[1])
            else:
                ys_left.append(p2[1])

        for p1, p2 in lines_right:
            if p1[0] == xmax:
                ys_right.append(p1[1])
            else:
                ys_right.append(p2[1])

        ys_left = sorted(list(set(ys_left)))
        ys_right = sorted(list(set(ys_right)))

        new_lines = []
        for i in range(1, len(ys_left)):
            new_lines.append(((xmin, ys_left[i - 1]), (xmin, ys_left[i])))
        for i in range(1, len(ys_right)):
            new_lines.append(((xmax, ys_right[i - 1]), (xmax, ys_right[i])))

        # remove the left and right line
        lines_with_borders = []
        for i, line in enumerate(lines):
            if not (i == idx_left or i == idx_right):
                lines_with_borders.append(line)

        lines_with_borders += new_lines

        ################################################################################
        # er zijn soms ook laagscheidingen die het maaiveld raken op een punt dat geen #
        # onderdeel is van het maaiveld, deze lijnen moeten opgesplitst worden in twee #
        # lijnen                                                                       #
        ################################################################################
        print("Toevoegen laagscheidingspunten op het maaiveld...")

        # vind eerst alle mogelijke polygonen, let op dat er ook overlappende polygonen zijn
        # die moeten we later weghalen

        polygons = find_polygons(lines_with_borders)
        total_pg = orient(unary_union(polygons), sign=-1)

        # bepaal de punten van de omhullende
        boundary = [
            (round(p[0], 3), round(p[1], 3))
            for p in list(zip(*total_pg.exterior.coords.xy))[:-1]
        ]
        # vind het linker punt van het maaiveld
        left = min([p[0] for p in boundary])
        topleft_point = sorted(
            [p for p in boundary if p[0] == left], key=lambda x: x[1]
        )[-1]

        # idem voor rechts
        right = max([p[0] for p in boundary])
        rightmost_point = sorted(
            [p for p in boundary if p[0] == right], key=lambda x: x[1]
        )[-1]

        # dit is waarschijnlijk overbodig maar toch even checken of we clockwise gaan
        if Polygon(boundary).exterior.is_ccw:
            boundary = boundary[::-1]

        # we weten het linker- en rechterpunt van het maaiveld, vindt de tussenliggende punten
        idx1 = boundary.index(topleft_point)
        idx2 = boundary.index(rightmost_point) + 1
        if idx1 > idx2:
            idx2 += len(boundary)
            mv = (boundary + boundary)[idx1:idx2]
        else:
            mv = boundary[idx1:idx2]

        # haal alle bestaande maaiveld lijnen weg
        filtered_lines = []
        xs_mv = [p[0] for p in mv]
        for p1, p2 in lines_with_borders:
            # skip de verticale lijnen aan de randen
            if p1[0] == p2[0] and (p1[0] == xmin or p1[0] == xmax):
                filtered_lines.append((p1, p2))
                continue

            if not p1 in mv or not p2 in mv:
                filtered_lines.append((p1, p2))

        # voeg de lijnen weer toe maar controleer ook of er snijpunten zijn met de overige lijnen
        mv_lijnen = []
        for i in range(1, len(mv)):
            p1 = mv[i - 1]
            p2 = mv[i]
            x1, y1 = p1
            x2, y2 = p2
            mv_lijn = [(p1, p2)]

            for j in range(len(filtered_lines)):
                # check of lijn i lijn j raakt (maar niet op het begin- of eindpunt)
                p3, p4 = filtered_lines[j]
                x3, y3 = p3
                x4, y4 = p4

                touching_point = None
                if x1 == x2:  # verticale lijn
                    if x3 == x1:
                        touching_point = (x3, y3)
                    elif x4 == x1:
                        touching_point = (x4, y4)
                else:  # niet verticale lijn
                    if x1 < x3 and x3 < x2:
                        yc = y1 + (x3 - x1) / (x2 - x1) * (y2 - y1)
                        if abs(yc - y3) < EPSILON:
                            touching_point = (x3, y3)
                    elif x1 < x4 and x4 < x2:
                        yc = y1 + (x4 - x1) / (x2 - x1) * (y2 - y1)
                        if abs(yc - y4) < EPSILON:
                            touching_point = (x4, y4)

                if touching_point is not None:
                    mv_lijn = [((x1, y1), touching_point), (touching_point, (x2, y2))]

            mv_lijnen += mv_lijn

        result.lines = filtered_lines + mv_lijnen

        return result

    def generate_polygons(self):
        print("Extraheren van de polygonen uit de losse lijnen...")
        self.polygons = find_polygons(self.lines, remove_overlapping=True)

    def to_stix(self, filename: str):
        if len(self.polygons) == 0:
            self.generate_polygons()

        levee = Levee()
        levee.soils.append(
            Soil(
                code="ongedefinieerd",
                yd=14.0,
                ys=10.0,
                c=2.0,
                phi=25.0,
                color="#808080",
            )
        )
        for poly in self.polygons:
            x, y = poly.exterior.xy
            levee.soilpolygons.append(
                SoilPolygon(soilcode="ongedefinieerd", points=zip(x, y))
            )
        levee.to_stix(filename)


def main(dxf_file):
    dxf_model = DXFModel.from_dxf(dxf_file)
    dxf_model.generate_polygons()
    dxf_model.to_stix(f"{dxf_file}.stix")

    if DEBUG:
        fig, ax = plt.subplots()
        for line in dxf_model.lines:
            x_values = [line[0][0], line[1][0]]
            y_values = [line[0][1], line[1][1]]
            ax.plot(x_values, y_values, color="b")
            ax.scatter(x_values, y_values, color="b")

        for i, poly in enumerate(dxf_model.polygons):
            x, y = poly.exterior.xy
            patch = mplPolygon(
                list(zip(x, y)),
                facecolor=generate_random_color(),
                alpha=0.5,
                edgecolor="black",
            )
            ax.add_patch(patch)

        plt.show()


if __name__ == "__main__":
    dxf_files = case_insensitive_glob(Path(DXF_PATH), ".dxf")
    for dxf_file in dxf_files:
        print(f"Bezig met DXF bestand '{dxf_file}'")
        main(dxf_file)
