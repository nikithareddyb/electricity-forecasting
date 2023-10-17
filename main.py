import yaml

with open("config.yml", "r") as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)

print(cfg["mysql"]["host"])
print(cfg["other"])

# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
#if __name__ == '__main__':
    # # print_hi('PyCharm')
    # # Create a configparser object
    # config = ConfigParser.RawConfigParser(allow_no_value=True)
    # # Read the configuration file
    # config.read('config.ini')
    # # Access configuration values
    # electricity_data_path = config.get('Datapaths', 'rawelectricitydata')
    # print(electricity_data_path)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
