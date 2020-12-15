from person_vehicle_monitoring.tools.line_utils import get_line_config, is_poi_with_in_poly, cross_res


def filter_vehicle(structures, equipment_id=None, debug=False):
    if debug:
        normal_line = [[(0, 0), (2000, 0), (2000, 2000), (0, 2000), "head"]]

    else:
        normal_line = get_line_config(equipment_id)
    if normal_line is None:
        return [], 'back'
    max_area = 0
    unique_structure = []
    line = normal_line[0]
    direction = line[-1]
    for structure in structures:
        print('output filter_vehicle structure')
        print(structure)
        classes_id = structure[0].decode()
        confidence = structure[1]
        poi = [structure[2][0], structure[2][1]]
        #  TODO  add vehicle type
        if classes_id in ['car', 'truck'] and confidence > 0.5:
            poly = [line[0], line[3], line[2], line[1]]
            if is_poi_with_in_poly(poi, poly) and cross_res(poi, poly):
                unique_structure.append(structure)

    return unique_structure, direction
