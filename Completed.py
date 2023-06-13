import bpy
import bmesh
from mathutils import Vector
import random
from scipy.spatial import Voronoi, Delaunay
import math

def create_voronoi(points, scale=1):
    # Создаем объект Voronoi на основе переданных точек
    vor = Voronoi(points)
    
    # Преобразуем вершины векторов в трехмерные координаты и масштабируем их
    vertices = [Vector((vert[0], vert[1], 0)) * scale for vert in vor.vertices]
    
    # Создаем список ребер, исключая те, у которых один из индексов вершин меньше нуля
    edges = [(edge[0], edge[1]) for edge in vor.ridge_vertices if edge[0] >= 0 and edge[1] >= 0]

    # Возвращаем список вершин и список ребер в результате выполнения функции
    return vertices, edges


def create_delaunay(points, scale=1):
    # Создаем объект Delaunay на основе переданных точек
    tri = Delaunay(points)
    
    # Преобразуем вершины треугольников в трехмерные координаты и масштабируем их
    vertices = [Vector((vert[0], vert[1], 0)) * scale for vert in tri.points]
    
    # Создаем список ребер путем объединения треугольников из tri.simplices
    edges = [(simp[0], simp[1]) for simp in tri.simplices] + [(simp[1], simp[2]) for simp in tri.simplices] + [(simp[2], simp[0]) for simp in tri.simplices]

    # Возвращаем список вершин и список ребер в результате выполнения функции
    return vertices, edges


def create_mesh_object(vertices, edges, width, name="Diagram", generation_type="voronoi"):
    # Создаем новую сетку и объект меша в Blender
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)

    # Создаем BMesh для работы с мешем
    bm = bmesh.new()
    bm_verts = [bm.verts.new(vert) for vert in vertices]

    # Создаем грани между вершинами ребер
    for edge in edges:
        road_vertices = create_road(vertices[edge[0]], vertices[edge[1]], width)
        bm_verts = [bm.verts.new(road_vert) for road_vert in road_vertices]
        bm.faces.new(bm_verts)

    # Подразделяем ребра для гладкости
    bmesh.ops.subdivide_edges(bm, edges=bm.edges, cuts=2, use_grid_fill=True)
    bmesh.ops.subdivide_edges(bm, edges=bm.edges, cuts=0, use_grid_fill=True)

    # Применяем BMesh к сетке меша
    bm.to_mesh(mesh)
    bm.free()

    # Возвращаем созданный объект меша
    return obj


def generate_points(number_of_points, min_coord, max_coord):
    points = []
    for i in range(number_of_points):
        x = random.uniform(min_coord.x, max_coord.x)
        y = random.uniform(min_coord.y, max_coord.y)
        points.append([x, y])
    return points



def create_road(start, end, width):
    # Вычисляем направление и нормаль для дороги
    direction = end - start
    normal = Vector((-direction.y, direction.x, 0)).normalized()
    offset = normal * (width / 2)

    # Создаем вершины для дороги
    vertices = [
        start - offset,
        start + offset,
        end + offset,
        end - offset
    ]

    # Возвращаем вершины дороги
    return vertices

def is_point_inside_area(point, min_coord, max_coord):
    # Проверяем, находится ли точка внутри заданной области
    return min_coord.x <= point.x <= max_coord.x and min_coord.y <= point.y <= max_coord.y

def clip_voronoi_edges_to_area(vertices, edges, min_coord, max_coord):
    new_edges = []
    for edge in edges:
        start = vertices[edge[0]]
        end = vertices[edge[1]]
        mid_point = start + (end - start) * 0.5 

        # Проверяем, находится ли середина ребра внутри области
        if is_point_inside_area(mid_point, min_coord, max_coord):
            new_edges.append(edge)
    
    # Возвращаем отфильтрованные ребра
    return new_edges


def house_intersects_houses(house_vertices, houses):
    house_segments = [
        (house_vertices[0], house_vertices[1]),
        (house_vertices[1], house_vertices[2]),
        (house_vertices[2], house_vertices[3]),
        (house_vertices[3], house_vertices[0])
    ]
    
    for existing_house in houses:
        existing_house_segments = [
            (existing_house[0], existing_house[1]),
            (existing_house[1], existing_house[2]),
            (existing_house[2], existing_house[3]),
            (existing_house[3], existing_house[0])
        ]
        
        for house_segment in house_segments:
            for existing_house_segment in existing_house_segments:
                if segments_intersect(house_segment[0], house_segment[1], existing_house_segment[0], existing_house_segment[1]):
                    return True
    return False


def generate_houses(vertices, edges, house_length, house_width, house_height, buffer_distance, min_distance_between_houses):
    houses = []

    # Проходимся по всем ребрам дорог
    for edge in edges:
        start = vertices[edge[0]]
        end = vertices[edge[1]]
        road_length = (end - start).length

        direction = (end - start).normalized()
        normal = Vector((-direction.y, direction.x, 0)).normalized()

        # Вычисляем общее пространство для дома и минимальное расстояние между домами
        total_house_space = house_length + min_distance_between_houses
        num_houses = int((road_length - 2 * buffer_distance) / total_house_space)

        # Проходимся по каждой стороне дороги (-1 и 1)
        for side in [-1, 1]: 
            offset = normal * (buffer_distance + house_width / 2) * side

            # Проверяем, что есть возможность разместить хотя бы один дом
            if num_houses > 0:
                segment_length = (road_length - 2 * buffer_distance) / num_houses

                # Проходимся по каждому дому внутри отрезка
                for i in range(num_houses):
                    house_start = start + direction * (buffer_distance + i * segment_length) + offset
                    house_end = house_start + direction * house_length

                    house_vertices = [
                        house_start,
                        house_end,
                        house_end + normal * house_width * side,
                        house_start + normal * house_width * side,
                        house_start + Vector((0, 0, house_height)),
                        house_end + Vector((0, 0, house_height)),
                        house_end + normal * house_width * side + Vector((0, 0, house_height)),
                        house_start + normal * house_width * side + Vector((0, 0, house_height))
                    ]

                    # Проверяем, не пересекается ли дом с дорогами и другими домами
                    if not house_intersects_roads(house_vertices, vertices, edges) and not house_intersects_houses(house_vertices, houses):
                        houses.append(house_vertices)

    # Возвращаем список домов
    return houses


def import_house_obj(filepath):
    # Импортируем OBJ-файл в сцену
    bpy.ops.import_scene.obj(filepath=filepath)

    # Выбираем импортированный объект и удаляем его из текущей коллекции
    imported_obj = bpy.context.selected_objects[0] 
    bpy.context.collection.objects.unlink(imported_obj)

    # Возвращаем импортированный объект
    return imported_obj


def add_house_instances_to_scene(houses, house_obj, texture_filepath):
    # Проходимся по каждому дому в списке
    for house_vertices in houses:
        # Вычисляем центр дома и его размеры
        house_center = sum((house_vertices[0], house_vertices[1], house_vertices[2], house_vertices[3]), Vector()) / 4
        house_dimensions = Vector((
            (house_vertices[1] - house_vertices[0]).length,
            (house_vertices[3] - house_vertices[0]).length,
            (house_vertices[4] - house_vertices[0]).length
        ))

        # Создаем экземпляр дома, масштабируем и позиционируем его
        instance = house_obj.copy()
        instance.scale = house_dimensions / 1000
        instance.location = house_center

        # Определяем направление и угол поворота для корректной ориентации дома
        direction = (house_vertices[1] - house_vertices[0]).normalized()
        angle = math.atan2(direction.y, direction.x)
        instance.rotation_euler = (math.pi/2, 0, angle)

        # Добавляем экземпляр дома в сцену
        bpy.context.collection.objects.link(instance)

        # Назначаем текстуру объекту
        set_texture_to_object(instance, texture_filepath)




def segments_intersect(p1, p2, q1, q2):
    # Функция для вычисления векторного произведения
    def cross(a, b):
        return a.x * b.y - a.y * b.x

    # Функция для проверки, лежит ли точка на отрезке
    def is_point_on_segment(p, q1, q2):
        return min(q1.x, q2.x) <= p.x <= max(q1.x, q2.x) and min(q1.y, q2.y) <= p.y <= max(q1.y, q2.y)

    r = p2 - p1
    s = q2 - q1
    denom = cross(r, s)

    if denom == 0:
        return False

    t = cross(q1 - p1, s) / denom
    u = cross(q1 - p1, r) / denom

    if 0 <= t <= 1 and 0 <= u <= 1:
        intersection = p1 + t * r
        return is_point_on_segment(intersection, p1, p2) and is_point_on_segment(intersection, q1, q2)

    return False


def house_intersects_roads(house_vertices, vertices, edges):
    # Создаем сегменты стен дома
    house_segments = [
        (house_vertices[0], house_vertices[1]),
        (house_vertices[1], house_vertices[2]),
        (house_vertices[2], house_vertices[3]),
        (house_vertices[3], house_vertices[0])
    ]

    # Проверяем пересечение с дорогами
    for edge in edges:
        road_start = vertices[edge[0]]
        road_end = vertices[edge[1]]

        for house_segment in house_segments:
            if segments_intersect(road_start, road_end, house_segment[0], house_segment[1]):
                return True

    return False


def import_tree_obj(filepath):
    # Импортируем OBJ-файл дерева в сцену
    bpy.ops.import_scene.obj(filepath=filepath)

    # Выбираем импортированный объект и удаляем его из текущей коллекции
    imported_obj = bpy.context.selected_objects[0] 
    bpy.context.collection.objects.unlink(imported_obj)

    # Возвращаем импортированный объект
    return imported_obj


def generate_tree_positions(num_trees, min_coord, max_coord, vertices, edges, houses, tree_radius=0.5, padding=0.2, max_attempts=100):
    tree_positions = []
    
    for i in range(num_trees):
        attempts = 0
        
        while attempts < max_attempts:
            # Генерируем случайную позицию для дерева в заданных границах
            x = random.uniform(min_coord.x, max_coord.x)
            y = random.uniform(min_coord.y, max_coord.y)
            tree_pos = Vector((x, y, 0))
            
            too_close = False
            
            # Проверяем, чтобы дерево не было слишком близко к дорогам
            for edge in edges:
                road_start = vertices[edge[0]]
                road_end = vertices[edge[1]]
                closest_point_on_road = closest_point_on_segment(tree_pos, road_start, road_end)
                distance_to_road = (closest_point_on_road - tree_pos).length
                if distance_to_road < tree_radius + padding:
                    too_close = True
                    break
                    
            # Проверяем, чтобы дерево не было слишком близко к домам
            if not too_close:
                for house in houses:
                    house_bottom_center = sum((house[0], house[1], house[2], house[3]), Vector()) / 4
                    distance_to_house = (house_bottom_center - tree_pos).length
                    if distance_to_house < tree_radius + padding:
                        too_close = True
                        break
                        
            # Если дерево находится на безопасном расстоянии от дорог и домов, добавляем его позицию
            if not too_close:
                tree_positions.append(tree_pos)
                break
            
            attempts += 1
            
    # Возвращаем список позиций деревьев
    return tree_positions


def closest_point_on_segment(point, segment_start, segment_end):
    # Находим ближайшую точку на отрезке к данной точке
    segment_dir = segment_end - segment_start
    segment_len_squared = segment_dir.length_squared
    t = max(0, min(1, ((point - segment_start).dot(segment_dir)) / segment_len_squared))
    return segment_start + t * segment_dir


def circle_intersects_rect(circle_center, circle_radius, rect_vertices):
    # Проверяем, пересекает ли окружность прямоугольник
    closest_x = max(rect_vertices[0].x, min(circle_center.x, rect_vertices[2].x))
    closest_y = max(rect_vertices[0].y, min(circle_center.y, rect_vertices[2].y))
    distance_x = circle_center.x - closest_x
    distance_y = circle_center.y - closest_y
    return (distance_x ** 2) + (distance_y ** 2) < (circle_radius ** 2)


def add_trees_to_scene(tree_positions, tree_object, texture_filepath, tree_scale=0.1):
    # Добавляем деревья в сцену
    for pos in tree_positions:
        obj = bpy.data.objects.new("Tree", tree_object.data)
        obj.location = pos
        obj.scale = (tree_scale, tree_scale, tree_scale)
        obj.rotation_euler = (math.pi/2, 0, 0)
        bpy.context.scene.collection.objects.link(obj)
        
        # Назначаем текстуру объекту
        set_texture_to_object(obj, texture_filepath)



def create_stone_material(name="StoneMaterial"):
    # Создаем материал для камня
    stone_material = bpy.data.materials.new(name=name)
    
    stone_material.use_nodes = True
    nodes = stone_material.node_tree.nodes
    
    # Удаляем все узлы из материала
    for node in nodes:
        nodes.remove(node)
    
    # Создаем узлы для текстуры кирпичей
    diffuse_shader = nodes.new(type='ShaderNodeBsdfDiffuse')
    diffuse_shader.location = (0, 0)

    mix_shader = nodes.new(type='ShaderNodeMixShader')
    mix_shader.location = (-300, 0)

    brick_texture = nodes.new(type='ShaderNodeTexBrick')
    brick_texture.location = (-600, 0)
    brick_texture.inputs['Color1'].default_value = (0.8, 0.4, 0.2, 1) 
    brick_texture.inputs['Color2'].default_value = (0.5, 0.25, 0.15, 1) 
    brick_texture.inputs['Scale'].default_value = 500.0 
    
    material_output = nodes.new(type='ShaderNodeOutputMaterial')
    material_output.location = (300, 0)
    
    # Соединяем узлы
    links = stone_material.node_tree.links
    link = links.new
    link(brick_texture.outputs["Color"], mix_shader.inputs[2])
    link(diffuse_shader.outputs["BSDF"], mix_shader.inputs[1])
    link(mix_shader.outputs["Shader"], material_output.inputs["Surface"])

    # Возвращаем созданный материал
    return stone_material

def set_texture_to_object(obj, texture_filepath):
    # Создаем материал с текстурой и назначаем его объекту
    mat = bpy.data.materials.new(name="Texture_Material")
    
    mat.use_nodes = True
    
    nodes = mat.node_tree.nodes
    
    # Удаляем все узлы из материала
    for node in nodes:
        nodes.remove(node)
    
    # Создаем узел для текстуры
    texture_node = nodes.new(type='ShaderNodeTexImage')
    texture_node.location = (0, 0)
    texture_node.image = bpy.data.images.load(texture_filepath)
    
    bsdf_node = nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf_node.location = (300, 0)
    
    material_output = nodes.new('ShaderNodeOutputMaterial')
    material_output.location = (600, 0)
    
    # Соединяем узлы
    links = mat.node_tree.links
    links.new(texture_node.outputs["Color"], bsdf_node.inputs["Base Color"])
    links.new(bsdf_node.outputs["BSDF"], material_output.inputs["Surface"])
    
    obj.data.materials[0] = mat
    obj.data.materials.append(mat)



# Удаляем все объекты в сцене
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Генерируем случайные точки
num_points = 20
min_coord = Vector((-20, -20, 0))
max_coord = Vector((20, 20, 0))
random_points = generate_points(num_points, min_coord, max_coord)

# Генерируем диаграмму Вороного или триангуляцию Делоне в зависимости от выбранного типа
generation_type = "voronoi"

if generation_type == "voronoi":
    vertices, edges = create_voronoi(random_points)
elif generation_type == "delaunay":
    vertices, edges = create_delaunay(random_points)
else:
    raise ValueError("Unsupported generation type")

# Обрезаем ребра диаграммы Вороного по границам области
edges = clip_voronoi_edges_to_area(vertices, edges, min_coord, max_coord)

# Создаем объект сетки дороги
road_width = 0.5
voronoi_obj = create_mesh_object(vertices, edges, road_width, generation_type)

# Генерируем дома
house_length = 2
house_width = 1
house_height = 2
buffer_distance = 0.1
min_distance_between_houses = 1.0
houses = generate_houses(vertices, edges, house_length, house_width, house_height, buffer_distance, min_distance_between_houses)

# Импортируем модель дома
house_obj = import_house_obj(r"C:\Users\SystemX\Desktop\GenerateRoadBlend\Models\house3.obj")

# Назначаем текстуру домам
house_texture_filepath = r"C:\Users\SystemX\Desktop\GenerateRoadBlend\Models\house3 Base Color.png"
add_house_instances_to_scene(houses, house_obj, house_texture_filepath)

# Генерируем позиции для деревьев
tree_radius = 0.3
num_trees = 50
tree_positions = generate_tree_positions(num_trees, min_coord, max_coord, vertices, edges, houses, tree_radius)

# Импортируем модель дерева
tree_obj = import_tree_obj(r"C:\Users\SystemX\Desktop\GenerateRoadBlend\Models\tree.obj")

# Назначаем текстуру деревьям
texture_filepath = r"C:\Users\SystemX\Desktop\GenerateRoadBlend\Models\Tree Base Color.png"
add_trees_to_scene(tree_positions, tree_obj, texture_filepath)

# Создаем материал для камня и назначаем его дороге
stone_material = create_stone_material()
voronoi_obj.data.materials.append(stone_material)

