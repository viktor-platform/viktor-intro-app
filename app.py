import json

from geopy.geocoders import Nominatim
import plotly.graph_objects as go
import plotly.express as px
from datetime import date
from pathlib import Path
from io import BytesIO
import pyproj as proj
import pandas as pd
import numpy as np
import math
import text

from viktor.utils import convert_word_to_pdf, render_jinja_template
from viktor.external.word import WordFileTag, WordFileImage, render_word_file
from viktor import ViktorController, File, progress_message
from viktor.core import Color
from viktor.parametrization import ViktorParametrization, Step, Text, GeoPolygonField, NumberField, LineBreak, \
    GeoPointField, TextField, MultiSelectField
from viktor.views import WebView, WebResult, ImageView, ImageResult, GeometryAndDataView, GeometryAndDataResult, PlotlyView, PlotlyResult, \
    MapView, MapResult, MapPolygon, PDFView, PDFResult, DataGroup, DataItem
from viktor.geometry import Group, SquareBeam, Material, Line, Point, Extrusion, \
    LinearPattern, circumference_is_clockwise, Vector, CircularExtrusion, GeoPoint, \
    BidirectionalPattern
from viktor.errors import UserError, InputViolation

#define concrete options as a df
def concrete_options():
    data = {
        'Name': ['C7/8 Concrete', 'C10 Concrete', 'C15 Concrete', 'C20 Concrete', 'C25 Concrete', 'C30 Concrete', 'C35 Concrete', 'C40 Concrete'],
        'Strength': [70,100,150,200,250,300,350,400],
        'CO2 per m^3': [510,514,528,548,579,617,664,720],
        'Cost per m^3':[200,210,220,230,240,260,280,300]
    }
    df = pd.DataFrame(data)
    return df

def convert_latlon_to_xy(lat, lon, optz=False):
    crs_wgs = proj.Proj(init='epsg:4326') # assuming you're using WGS84 geographic
    crs_bng = proj.Proj(init='epsg:27700') # use a locally appropriate projected CRS

    # then cast your geographic coordinate pair to the projected system
    x_coord, y_coord = proj.transform(crs_wgs, crs_bng, lon, lat)
    if optz==True:
        print(x_coord, y_coord)
        return [x_coord, y_coord, 0]
    else:
        return [x_coord, y_coord]

def convert_xy_to_latlon(x, y, optz=False):
    crs_wgs = proj.Proj(init='epsg:4326') # assuming you're using WGS84 geographic
    crs_bng = proj.Proj(init='epsg:27700') # use a locally appropriate projected CRS

    # then cast your geographic coordinate pair to the projected system
    lon, lat = proj.transform(crs_bng, crs_wgs, x, y)
    if optz==True:
        return [lat, lon, 0]
    else:
        return [lat, lon]

def validate_maps_step(params, **kwargs):
    violations =[]
    if params.step_2.building_area is None:
        violations.append(InputViolation("The terrain input cannot be empty", fields=['step_2.building_area']))
    if params.step_2.building_point is None:
            violations.append(InputViolation("The building needs a point location", fields=['step_2.building_point']))
    if violations:
        raise UserError("Cannot make 3D Model", input_violations=violations)

class Parametrization(ViktorParametrization):
    #Intro parameters
    step_1 = Step("Introduction", views=['lets_do_this'])
    step_1.intro_text = Text(text.text1)

    #Maps parameters
    step_2 = Step("Maps", views=["get_map_view"], on_next=validate_maps_step)
    step_2.text = Text(text.text2)
    step_2.building_area = GeoPolygonField('**Step 1:** Draw the terrain for the building.')
    step_2.price_per_square_meter = NumberField('**Step 2:** Provide the price of the ground', min=1, default=50, flex=60)
    step_2.building_point = GeoPointField('**Step 3:** Place the building inside the terrain')
    step_2.building_angle = NumberField("**Step 4:** rotate the building", default=0, min = 0, max=360, variant='slider', flex=60)
    
    #3D models parameters
    step_3 = Step("3D Models", views=["get_building_model", "get_floor_plan"])
    step_3.intro = Text(text.text3)
    step_3.building_length = NumberField("Building length", min=5, max=100, step=5, default=30)
    step_3.building_width = NumberField("Building width", min=5, max=100, step=5, default=40)
    step_3.lb = LineBreak()
    step_3.number_floors = NumberField("how many floors", variant='slider', min=10, max=40, default=25)
    step_3.basement_floors = NumberField("Basement floors", default=3, min=1, max=7)
    
    #data parameters
    step_4 = Step("Visualise Data", views=["show_plotly"])
    step_4.text = Text(text.text5)
    step_4.concrete_choice = MultiSelectField("Select a cement type", options=concrete_options()['Name'].tolist(), default=concrete_options()['Name'].tolist(), flex=50)
    
    #integrations parameters
    step_5 = Step("Integrations", views=["show_cool_integrations"])
    step_5.integrations_text = Text(text.text4)

    #reporting parameters
    step_6 = Step("Reporting", views=["generate_report"])
    step_6.report_text =Text(text.text6)
    step_6.user_name = TextField("**Step 1:** Enter your name here", flex=50)
    step_6.lb = LineBreak()
    step_6.user_thoughts = TextField("**Step 2:** What did you think of this app?", flex=50)
    step_6.extra_text = Text("**Step 3:** press the update button to see your report ➡️")

    #whats next parameters
    step_7 = Step("What's Next", views=["whats_next"])


class Controller(ViktorController):
    label = 'My Entity Type'
    parametrization = Parametrization

    # intro slide
    @ImageView("", duration_guess=1)
    def lets_do_this(self, params, **kwargs):
        file = File.from_path(Path(__file__).parent / 'intro_app_banner.png')
        return ImageResult(file)
    
    # map view for land poly and building 
    @MapView("Map View", duration_guess=1)
    def get_map_view(self, params, **kwargs):
        building_area = params.step_2.building_area
        building_point = params.step_2.building_point
        price_square = params.step_2.price_per_square_meter
        features = []
        if building_area:
            land_polygon, land_centroid = self.conversion_centroid(building_area.points)
            features.append(MapPolygon.from_geo_polygon(building_area, color=Color.green(), 
                                                        description=f"Cost of Land: €{np.round(self.calculate_polygon_area(land_polygon)*price_square)}",
                                                        title=f'Total Area: {np.round(self.calculate_polygon_area(land_polygon),0)} m^2'))

        if building_point:
            profile, converted = self.point_to_polygon(params)
            features.append(MapPolygon(profile))
        return MapResult(features)
    
    #convert a point to a square for building
    @staticmethod
    def point_to_polygon(params):
        half_length = params.step_3.building_length/2
        half_width = params.step_3.building_width/2
        poly_angle = params.step_2.building_angle*math.pi/180
        map_point = params.step_2.building_point   
        centroid = convert_latlon_to_xy(map_point.lat, map_point.lon)                                    
        converted_points = [[-half_length, half_width],
            [half_length, half_width],
            [half_length, -half_width],
            [-half_length, -half_width]] 
        profile = []
        for point in converted_points:
            x, y = point[0] + centroid[0], point[1] + centroid[1]
            cx, cy = centroid
            rotated_x = cx + (x - cx) * math.cos(poly_angle) - (y - cy) * math.sin(poly_angle)
            rotated_y = cy + (x - cx) * math.sin(poly_angle) + (y - cy) * math.cos(poly_angle)
            profile_point = convert_xy_to_latlon(rotated_x, rotated_y)                                         
            profile.append(GeoPoint(profile_point[0], profile_point[1]))
        return profile, converted_points
            
    #geometry of building and some preliminary data
    @GeometryAndDataView("Building Model", duration_guess=1)
    def get_building_model(self, params, **kwargs):
        building_area = params.step_2.building_area
        number_floors = params.step_3.number_floors
        building_angle = params.step_2.building_angle
        building_width = params.step_3.building_width
        building_length = params.step_3.building_length
        basement_floors = params.step_3.basement_floors

        grass = Material("grass", color=Color(r=111, g=166,b=31), opacity=0.3)
        building, building_centroid = self.building_model(params)
        land_polygon, land_centroid = self.conversion_centroid(building_area.points)
        land = self.shape_extrusion(land_polygon, Line(Point(0,0,-2*number_floors),Point(0,0,0)), material=grass)
        building, building_centroid = self.building_model(params)
        centroid_difference = building_centroid - land_centroid
        building.translate([centroid_difference[0], centroid_difference[1], 0])
        land.rotate(-building_angle*math.pi/180, Vector(0,0,1), point=(centroid_difference[0], centroid_difference[1], 0))


        features = Group([land, building])
        # features.translate([building_centroid[0], building_centroid[1], 0])
        data = DataGroup(
            DataItem("Height", number_floors*4 + 8,suffix='m' ),
            DataItem("Surface Area", building_width*building_length*number_floors, suffix='m^2'),
            DataItem("Basement Area", building_width*building_length*basement_floors, suffix="m^2"),
            DataItem("Pile Depth", 2*number_floors, suffix='m')
        )
        return GeometryAndDataResult(features, data)
    
    #the model that uses the params as inputs
    def building_model(self, params, **kwargs):
        map_building, building_polygon = self.point_to_polygon(params)
        map_point = params.step_2.building_point
        building_centroid = convert_latlon_to_xy(map_point.lat, map_point.lon)
        building_length = params.step_3.building_length
        building_width = params.step_3.building_width
        basement_floors = params.step_3.basement_floors
        number_floors = params.step_3.number_floors
        #hard params
        opacity=1
        glass_opacity = 0.25
        floor_height = 4
        columns_amount_length = int(building_length/10)
        columns_amount_width = int(building_width/10)
        #Materials
        glass = Material("Glass", color=Color(18, 20, 255), opacity=glass_opacity)
        facade = Material("facade", color=Color(255, 252, 245), opacity=opacity)
        floor_material = Material("internals", color=Color(255, 252, 245), opacity=opacity)
        basement = Material("Concrete basement", color=Color(255, 252, 245), opacity=opacity)
        #basic elements
        building_base = self.shape_extrusion(building_polygon,
                                                     Line(Point(0,0,0),Point(0,0,floor_height)),
                                                     material=facade,
                                                     scaling_vector=Vector(1.03, 1.03, 1.03)
                                                     )
        windows = self.shape_extrusion(building_polygon,
                                                     Line(Point(0,0,floor_height),Point(0,0,floor_height*number_floors)),
                                                     material=glass,
                                                     scaling_vector=Vector(1, 1, 1)
                                                     )
        roof = self.shape_extrusion(building_polygon,
                                                     Line(Point(0,0,0),Point(0,0,2)),
                                                     material=facade,
                                                     scaling_vector=Vector(1.03, 1.03, 1.03)
                                                     )
        roof.translate([0,0,floor_height*number_floors])
        flooring = self.shape_extrusion(building_polygon,
                                                      Line(Point(0,0,2*floor_height),Point(0,0,2*floor_height+0.5)),
                                                      material=facade, 
                                                      scaling_vector=Vector(1.001, 1.001, 1.001)
                                                      )
        basement = self.shape_extrusion(building_polygon,
                                                     Line(Point(0,0,-basement_floors*floor_height),Point(0,0,0)),
                                                     material=basement,
                                                     scaling_vector=Vector(0.99, 0.99, 0.99)
                                                     )
        top_feature = self.shape_extrusion(building_polygon,
                                                     Line(Point(0,0,0),Point(0,0,4)),
                                                     material=facade,
                                                     scaling_vector=Vector(0.8, 0.8, 1)
                                                     )
        top_feature.translate([0,0,floor_height*number_floors+2])
        other_top_feature = self.shape_extrusion(building_polygon,
                                                     Line(Point(0,0,0),Point(0,0,1)),
                                                     material=facade,
                                                     scaling_vector=Vector(1, 1, 1)
                                                     )
        other_top_feature.translate([0,0,floor_height*number_floors+5])

        elevator_shaft = SquareBeam(length_x=8, length_y=8, length_z=number_floors*floor_height, material=facade)
        elevator_shaft.translate([0, 0, number_floors*floor_height/2+2])
        
        all_floors = LinearPattern(flooring, direction=[0, 0, 1], number_of_elements=number_floors-1+basement_floors, spacing=floor_height)
        all_floors.translate([0,0,-(basement_floors+1)*floor_height])
        #grid elements
        pillars = Group([])
        for point in building_polygon:
            pillar = SquareBeam(length_x=1, length_y=1, length_z=number_floors*floor_height, material=facade)
            pillar.translate([point[0], point[1], number_floors*floor_height/2+2])
            pillars.add(pillar)
            
        line = Line(Point(0,0,0),Point(0,0,number_floors*(2+floor_height)))
        col = CircularExtrusion(diameter=0.5, line=line, material=facade)

        columns = BidirectionalPattern(col, 
                                       direction_1= [1,0,0], 
                                       direction_2 = [0,1,0], 
                                       number_of_elements_1=columns_amount_length+2,
                                       number_of_elements_2=columns_amount_width+2,
                                       spacing_1=building_length/(columns_amount_length+1),
                                       spacing_2=building_width/(columns_amount_width+1))
        columns.translate([-building_length/2,-building_width/2,-2*number_floors])

        model = Group([basement, 
                       pillars, 
                       columns, 
                       building_base, 
                       windows, 
                       roof, 
                       all_floors, 
                       elevator_shaft,
                       top_feature,
                       other_top_feature
                       ])

        return model, building_centroid

    #plotly to display floor plan of an arbitrary floor
    @PlotlyView("Floor Plan", duration_guess=1)
    def get_floor_plan(self, params, **kwargs):
        fig = self.make_floor_plan(params)
        return PlotlyResult(fig.to_json())
    
    # make the floor plan in plotly
    @staticmethod
    def make_floor_plan(params):
        building_length = params.step_3.building_length
        building_width = params.step_3.building_width

        fig = go.Figure()
        #add outline & pillars
        exeterior_points = np.array([
            [0, 0],
            [building_length, 0],
            [building_length, building_width],
            [0, building_width],
            [0, 0],
            ])
        fig.add_trace(go.Scatter(x=exeterior_points[:,0], 
                                 y=exeterior_points[:,1], 
                                 mode='lines+markers', 
                                 line=dict(dash='dash', color='rgb(0, 0, 200)'),
                                 marker=dict(size=20,
                                             color='rgb(128, 128, 128)',
                                             symbol='square')
                                 ))
        # add elevator shaft
        elevator_points = np.array([
            [building_length/2-4, building_width/2-4],
            [building_length/2+4, building_width/2-4],
            [building_length/2+4, building_width/2+4],
            [building_length/2-4, building_width/2+4],
            [building_length/2-4, building_width/2-4],
            ])
        fig.add_trace(go.Scatter(x=elevator_points[:,0], 
                                 y=elevator_points[:,1], 
                                 mode='lines', 
                                 line=dict(color='rgb(128, 128, 128)', width=10),
                                ))

        #add columns
        x=np.linspace(0, building_length, int(building_length/10)+2)
        y=np.linspace(0, building_width, int(building_width/10)+2)
        for i in x:
            for j in y:
                if building_width/2-4 <= j <= building_width/2+4 and building_length/2-4 <= i <= building_length/2+4: 
                    continue
                else:
                    fig.add_trace(go.Scatter(x=[i],
                                             y=[j],
                                             mode='markers',
                                             marker=dict(size=10,
                                             color='rgb(128, 128, 128)',
                                             symbol='circle')
                                             ))
        fig.update_layout(
            showlegend=False,
            title='Building Floor Plan',
            yaxis=dict(scaleanchor='x', scaleratio=1),
            )
        fig.update_layout(
            plot_bgcolor='white'
        )
        fig.update_xaxes(
            mirror=True,
            ticks='outside',
            showline=True,
            gridcolor='lightgrey'
        )
        fig.update_yaxes(
            mirror=True,
            ticks='outside',
            showline=True,
            gridcolor='lightgrey'
        )
        return fig

    #extrude shapes from map points polygon
    @staticmethod
    def shape_extrusion(translated_coordinates, extrusion_line, material, scaling_vector=Vector(1,1,1)):
        profile = []
        for point in translated_coordinates:
            profile.append(Point(point[0], point[1], 0))
        profile.append(profile[0])
        if circumference_is_clockwise(profile) is not True:
            profile.reverse()
        extrusion = Extrusion(profile, extrusion_line, material=material)
        extrusion.scale(scaling_vector)
        return extrusion

    #calculate and convert points to xyz 
    @staticmethod
    def conversion_centroid(shape_points_from_map):                        
        converted_coordinates = [convert_latlon_to_xy(point.lat, point.lon) for point in shape_points_from_map]
        centroid = np.mean(converted_coordinates, axis=0)
        translated_coordinates = [(coordinate - centroid) for coordinate in converted_coordinates]
        return translated_coordinates, centroid

    # graph of CO2 cost of building with concrete
    @PlotlyView("Concrete Analysis", duration_guess=1)
    def show_plotly(self, params, **kwargs):
        fig = self.make_co2_plot(params)
        return PlotlyResult(fig.to_json())

    #make graph for the co2 trade off
    @staticmethod
    def make_co2_plot(params, **kwargs):
        data = concrete_options()
        options = params.step_4.concrete_choice

        for entry in data['Name']:
            if entry not in options:
                data = data.drop(data[data['Name']==entry].index)

        fig = go.Figure()
        fig.add_trace(go.Bar(x=data['Name'],
                y=data['Cost per m^3'],
                name='Cost €/m³',
                text=data['Cost per m^3'],
                marker_color='rgb(26, 118, 255)',
                )
        )
        fig.add_trace(go.Bar(x=data['Name'],
                y=data['CO2 per m^3'],
                name='CO2 Kg/m³',
                text=data['CO2 per m^3'],
                marker_color='indianred',
                )
        )
        fig.add_trace(go.Bar(x=data['Name'],
                y=data['Strength'],
                name='Strength Pa x 10E5',
                text=data['Strength'],
                marker_color='lightsalmon',
                )
        )
        fig.update_layout(title="Concrete comparison", 
                          barmode='group', 
                          yaxis=dict(visible=False),
                          xaxis_title="Concrete Type")
    
        return fig

    #display report
    @PDFView("Final report", duration_guess=10)
    def generate_report(self, params, **kwargs):
        land_polygon, land_centroid = self.conversion_centroid(params.step_2.building_area.points)
        building_point = params.step_2.building_point
        user_name = params.step_6.user_name
        user_thoughts = params.step_6.user_thoughts
        price_per_square_meter = params.step_2.price_per_square_meter
        number_floors = params.step_3.number_floors
        building_width =params.step_3.building_width
        building_length = params.step_3.building_length
        basement_floors = params.step_3.basement_floors

        geolocator = Nominatim(user_agent="ViktorSearch")
        location = geolocator.reverse(f"{building_point.lat},{building_point.lon}")
        address = location.raw['address']

        components = []

        fig = self.make_co2_plot(params)
        co2_plotly_fig = BytesIO(fig.to_image(format="png", scale=2))
        fig = self.make_floor_plan(params)
        floor_plans = BytesIO(fig.to_image(format="png", scale=2))

        
        progress_message(message="Preparing data and images...")

        data = {
            "user_name": user_name,
            "today_date": date.today().strftime("%d/%m/%Y"),
            "user_thoughts":user_thoughts,
            "country": address.get('country', ''),
            "city": address.get('city', ''),
            "price_psm" : price_per_square_meter,
            "prop_price": np.round(price_per_square_meter*self.calculate_polygon_area(land_polygon), 2),
            "height": number_floors*4 + 8,
            "width": building_width,
            "length": building_length,
            "floors": number_floors,
            "basement_floors": basement_floors,
            "surface_area": building_width*building_length*number_floors,
            "basement_area": building_width*building_length*basement_floors,
            "pile_depth": 2*number_floors,
        }

        images = {
            "co2_graph": co2_plotly_fig,
            "floor_plan": floor_plans,
        }

        # make tags
        for tag, value in data.items():
            components.append(WordFileTag(tag, value))
        for key, image in images.items():
            components.append(WordFileImage(image, key, width=432))

        # Get path to template and render word file
        template_path = Path(__file__).parent / "report_template_1.docx"
        with open(template_path, 'rb') as template:
            word_file = render_word_file(template, components)
            pdf_file = convert_word_to_pdf(word_file.open_binary())

        progress_message(message="Rendering the report...")

        return PDFResult(file=pdf_file)


    #display a simple image with integrations to inform the user
    @ImageView("Integrations", duration_guess=1)
    def show_cool_integrations(self, params, **kwargs):
        file = File.from_path(Path(__file__).parent / 'integrations.svg')
        return ImageResult(file)


    # Show final step
    @WebView("What's next?", duration_guess=1)
    def whats_next(self, **kwargs):
        """Initiates the process of rendering the "What's next?" tab."""
        html_path = Path(__file__).parent / "info_page" / "html_template.html"
        input_path = Path(__file__).parent / "info_page" / "info_input.json"
        with input_path.open() as f:
            input_path = json.load(f)
        with open(html_path, 'rb') as template:
            html_file = render_jinja_template(template, input_path)
        html_path = Path(__file__).parent / "info_page" / "html_sample.html"
        with open(html_path, 'w') as sample:
            sample.write(html_file.getvalue())
        return WebResult(html=html_file.getvalue())
    
    #Use shoelace method to determine the area of any polygon
    @staticmethod
    def calculate_polygon_area(vertices):
        n = len(vertices)
        if n < 3:
            return 0
        vertices.append(vertices[0])
        # Apply the shoelace formula
        area = 0
        for i in range(n):
            area += (vertices[i][0] * vertices[i + 1][1]) - (vertices[i + 1][0] * vertices[i][1])
        # Take the absolute value and divide by 2
        area = abs(area) / 2
    
        return area
