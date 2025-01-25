import ezdxf
from typing import List, Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as mplPolygon
import numpy as np


from shapely.geometry import LineString, Polygon, MultiPolygon
from shapely.ops import polygonize, unary_union


DXF_FILE = "data/test.dxf"
DEBUG = False


def generate_random_color():
    """Generates a random RGB color as a tuple."""
    return tuple(np.random.rand(3))


class DXFPoint:
    def __init__(self, x: float, y: float):
        self.x = round(x, 2)
        self.y = round(y, 2)

    def to_tuple(self) -> Tuple[float, float]:
        return self.x, self.y


class DXFLine:
    def __init__(self, start: DXFPoint, end: DXFPoint):
        self.start = start
        self.end = end

    def to_shapely(self) -> LineString:
        return LineString([self.start.to_tuple(), self.end.to_tuple()])

    @property
    def xmin(self) -> float:
        pass

    @property
    def xmax(self) -> float:
        pass


class DXFPolyLine:
    def __init__(self, id: str, lines: List[DXFLine]):
        self.id = id
        self.lines = lines


class DXFModel:
    def __init__(self):
        self.lines = []
        self.lwpolylines = []
        self.polylines = []

    @classmethod
    def from_dxf(cls, filename: str) -> "DXFModel":
        result = DXFModel()

        doc = ezdxf.readfile(filename)
        msp = doc.modelspace()

        lines = []

        # lees de polylijnen in
        for polyline in msp.query("LWPOLYLINE"):
            # print(f"Polyline ID: {")
            id = polyline.dxf.handle
            points = [DXFPoint(float(p[0]), float(p[1])) for p in polyline.vertices()]
            for i in range(1, len(points)):
                p1 = points[i - 1]
                p2 = points[i]
                lines.append(DXFLine(start=p1, end=p2))

            # result.polylines.append(DXFPolyLine(id=id, lines=lines))

        for line in msp.query("LINE"):
            lines.append(
                DXFLine(
                    start=DXFPoint(line.dxf.start[0], line.dxf.start[1]),
                    end=DXFPoint(line.dxf.end[0], line.dxf.end[1]),
                )
            )

        # krijg de omhullende
        try:
            linestrings = [l.to_shapely() for l in lines]
            geom_collection = polygonize(linestrings)
            # de omhullende is de polygoon met het grootste oppervlak
            pg = sorted(geom_collection, key=lambda x: x.area)[-1]
        except Exception as e:
            raise ValueError(
                "Het lijkt erop dat deze DXF anders is opgesteld dan verwacht (drie losse lijnen die de linker- rechter- en top geometrie voorstellen en verder losse lijnen). Check de opmaak."
            )

        if DEBUG:
            fig, ax = plt.subplots()
            x, y = pg.exterior.xy
            ax.plot(x, y)
            plt.show()

        # bepaal de linker- en rechter limiet en rondt af op 2 decimalen
        xmin, _, xmax, _ = pg.bounds
        xmin = round(xmin, 2)
        xmax = round(xmax, 2)

        if DEBUG:
            print(f"xmin = {xmin}, xmax = {xmax}")

        ## Bij het huidige DXF format bestaan de grondlagen uit losse lijnen
        ## Er zijn 3 polylijnen aanwezig, te weten
        ## - het maaiveld
        ## - de linker limiet van de geometrie
        ## - de rechter limiet van de geometrie
        ## om hier de geometrie uit te halen moet het volgnd
        ## verwijder de lijn(en) die de linker en rechterlimiet van de geometrie vormen
        # let op dit werkt alleen als de punten op de linker- en rechterlimiet dezelfde
        # x coordinaat hebben, oftewel loodrecht op de y-as lopen
        final_lines = []
        for line in lines:
            skip = line.start.x == line.end.x and line.start.x == xmax
            skip |= line.start.x == line.end.x and line.start.x == xmin
            if not skip:
                final_lines.append(line)

        ## genereer de lijnstukken voor de linker- en rechterzijde
        # haal alle z coordinaten op voor xmin en xmax
        left_points = []
        right_points = []
        for l in final_lines:
            if l.start.x == xmin:
                left_points.append(l.start.y)
            if l.end.x == xmin:
                left_points.append(l.end.y)
            if l.start.x == xmax:
                right_points.append(l.start.y)
            if l.end.x == xmax:
                right_points.append(l.end.y)

        left_points = sorted(list(set(left_points)))
        right_points = sorted(list(set(right_points)))

        # genereer deze lijnen
        for i in range(1, len(left_points)):
            final_lines.append(
                DXFLine(
                    start=DXFPoint(x=xmin, y=left_points[i - 1]),
                    end=DXFPoint(x=xmin, y=left_points[i]),
                )
            )

        for i in range(1, len(right_points)):
            final_lines.append(
                DXFLine(
                    start=DXFPoint(x=xmax, y=right_points[i - 1]),
                    end=DXFPoint(x=xmax, y=right_points[i]),
                )
            )

        # maak een lijst van alle lijnen die de omhullende van de geometrie vormen
        p1, exterior_lines = None, []
        for x, y in zip(*pg.exterior.xy):
            p2 = (x, y)
            if p1 is not None:
                exterior_lines.append([(p1, p2)])
            p1 = p2

        # itereer over alle lijnen, als ze geen onderdeel zijn van de exterior lijnen
        # dan moet er gecontroleerd worden of ze een snijpunt hebben met de exterior lijn
        # wat niet het begin- of eindpunt van die exterior lijn is
        # zo ja dan moet die exterior lijn in twee stukken verdeeld worden

        print(exterior_lines)

        i = 1

        result.lines = final_lines
        return result

    def plot(self):
        fig, ax = plt.subplots()
        for line in self.lines:
            x_values = [line.start.x, line.end.x]
            y_values = [line.start.y, line.end.y]
            ax.plot(x_values, y_values, color="b")
            ax.scatter(x_values, y_values, color="b")

        for pline in self.polylines:
            for line in pline.lines:
                x_values = [line.start.x, line.end.x]
                y_values = [line.start.y, line.end.y]
                ax.plot(x_values, y_values, color="r")
                ax.scatter(x_values, y_values, color="r")

        # plot lines and polyline in different colors
        plt.show()

    def lines_to_shapely(self):
        result = []
        for line in self.lines:
            result.append(
                LineString([(line.start.x, line.start.y), (line.end.x, line.end.y)])
            )
        for plline in self.polylines:
            for line in plline.lines:
                result.append(
                    LineString([(line.start.x, line.start.y), (line.end.x, line.end.y)])
                )
        return result


# def extract_all_polygons_from_lines_graph_based(lines):
#     """
#     Extracts all polygons from a collection of lines by traversing the line graph.

#     Args:
#       lines: A list of shapely LineString objects.

#     Returns:
#       A list of shapely Polygon objects.
#     """

#     # 1. Build a graph representation of the lines
#     graph = {}
#     for line in lines:
#         start, end = line.coords[0], line.coords[-1]
#         if start not in graph:
#             graph[start] = []
#         graph[start].append((end, line))
#         if end not in graph:
#             graph[end] = []
#         graph[end].append((start, line))

#     # 2. Depth-First Search (DFS) to find closed paths
#     polygons = []
#     visited = set()

#     def dfs(start_point, current_path):
#         visited.add(start_point)
#         for neighbor, line in graph[start_point]:
#             if neighbor not in visited:
#                 new_path = current_path + [line]
#                 if (
#                     neighbor in graph and start_point in graph[neighbor]
#                 ):  # Check for closure
#                     polygons.append(
#                         Polygon(
#                             list(coord for line in new_path for coord in line.coords)
#                         )
#                     )
#                 else:
#                     dfs(neighbor, new_path)

#     # 3. Start DFS from each point
#     for start_point in graph:
#         if start_point not in visited:
#             dfs(start_point, [])

#     return polygons


def main():
    dxf_model = DXFModel.from_dxf(DXF_FILE)
    dxf_model.plot()

    lines = dxf_model.lines_to_shapely()
    polygons = list(polygonize(lines))

    fig, ax = plt.subplots()
    for i, poly in enumerate(polygons):
        x, y = poly.exterior.xy
        patch = mplPolygon(
            list(zip(x, y)),
            facecolor=generate_random_color(),
            alpha=0.5,
            edgecolor="black",
        )
        ax.add_patch(patch)

    ax.autoscale()
    plt.show()

    print(len(polygons))


if __name__ == "__main__":
    main()
