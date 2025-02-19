import ezdxf
from typing import List, Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as mplPolygon
import numpy as np


from shapely.geometry import LineString, Polygon, MultiPolygon
from shapely.ops import polygonize, unary_union
from leveelogic.objects.levee import Levee
from leveelogic.objects.soilpolygon import SoilPolygon
from leveelogic.objects.soil import Soil


DXF_FILE = "data/test_correctie.dxf"
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

        result.lines = final_lines
        return result

    def plot(self):
        _, ax = plt.subplots()
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
        polygons = self.to_polygons()
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

    def to_polygons(self):
        lines = self.lines_to_shapely()
        polygons = list(polygonize(lines))
        return polygons

    def to_stix(self, filename: str):
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
        polygons = self.to_polygons()
        for i, poly in enumerate(polygons):
            x, y = poly.exterior.xy
            levee.soilpolygons.append(
                SoilPolygon(soilcode="ongedefinieerd", points=zip(x, y))
            )
        levee.to_stix(filename)


def main():
    dxf_model = DXFModel.from_dxf(DXF_FILE)
    # dxf_model.plot()
    dxf_model.to_stix(f"{DXF_FILE}.stix")


if __name__ == "__main__":
    main()
