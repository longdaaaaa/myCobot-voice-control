from pymycobot import MyCobot320Socket

# 机器人连接参数
ROBOT_IP = "192.168.43.94"
ROBOT_PORT = 9000
mc = MyCobot320Socket(ROBOT_IP, ROBOT_PORT)

mc.focus_all_servos()