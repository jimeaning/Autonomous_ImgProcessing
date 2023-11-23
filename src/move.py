import rospy
from geometry_msgs.msg import Twist
import sys, select ,os

if os.name == 'nt':
    import msvcrt,time
else:
    import tty, termios
    
""""
nt = windows NT 를 의미함
리눅스 같은 경우에는 tty , termios 가 터미널 I/O와 설정 조작을 담당
윈도우 같은 경우에는 msvcrt가 담당한다.

"""
def move():
    rospy.init_node('autonomous_move')
    pub = rospy.Publisher('cmd_vel', Twist, queue_size = 10)
    
    twist = Twist()

    print("Lets move your robot")
    speed = 0.11
    distance = 5
    isForward = 1

    if(isForward):
        twist.linear.x = abs(speed)
    else:
        twist.linear.x = -abs(speed)
        
    twist.linear.y = twist.linear.z = 0
    twist.angular.x = twist.angular.y = twist.angular.z = 0
    
    

    while not rospy.is_shutdown():
        twist.linear.x = 0.22
        pub.publish(twist)
    if rospy.is_shutdown == 0:
        print("rospy is shutdonw")


if __name__ == '__main__':
    try:
        # Testing our function
        move()
    except rospy.ROSInterruptException: pass
    
    finally:
        twist = Twist()
        twist.linear.x = 0.0; twist.linear.y = 0.0; twist.linear.z = 0.0
        twist.angular.x = 0.0; twist.angular.y = 0.0; twist.angular.z = 0.0
        pub.publish(twist)
     
    if os.name != 'nt':
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    
    