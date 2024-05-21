

def convert_json_to_python(data):
    if isinstance(data, dict):
        for key, value in data.items():
            if value == "null":
                data[key] = None
            elif value == "true":
                data[key] = True
            elif value == "false":
                data[key] = False
            else:
                data[key] = convert_json_to_python(value)
    elif isinstance(data, list):
        for i in range(len(data)):
            if data[i] == "null":
                data[i] = None
            elif data[i] == "true":
                data[i] = True
            elif data[i] == "false":
                data[i] = False
            else:
                data[i] = convert_json_to_python(data[i])
    return data