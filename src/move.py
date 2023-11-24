import rospy
from geometry_msgs.msg import Twist
import sys, select ,os, threading

if os.name == 'nt':
    import msvcrt,time
else:
    import tty, termios, time
    
""""
nt = windows NT 를 의미함
리눅스 같은 경우에는 tty , termios 가 터미널 I/O와 설정 조작을 담당
윈도우 같은 경우에는 msvcrt가 담당한다.

"""


def getKey():
    
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''
    return key


def stop_motion():
    twist.linear.x= 0

def startTimer():
    global count, speed, green_flag, timer
    count += 1
    print("Count")
    timer = threading.Timer(1, startTimer)
    timer.start()
    
    
    if green_flag == 1:
        speed = 0.22
        print("초록불임")
        timer.cancel()
    
    if count == 5:
        speed = 0
        count =0
        green_flag = 0
        print("타이머를 멈춘다")
        
        timer.cancel()
        
        
if __name__ == '__main__':
    try:
        # Testing our function
        pub = rospy.Publisher('cmd_vel', Twist, queue_size = 10)
        rospy.init_node('autonomous_move')
        
        
        twist = Twist()

        msg = """
        
        Lets move your robot
        
        q = 신호등 발견
        w = 빨간불 이다.
        e = 초록불 이다.
        z = 시작한다. 
        m = 종료
        """
        print(msg)
        twist.linear.x = twist.linear.y = twist.linear.z = 0
        twist.angular.x = twist.angular.y = twist.angular.z = 0
        
        count = 0
        speed = 0
        current_speed = 0
        green_flag = 0
        pub.publish(twist)
        
        while not rospy.is_shutdown():
            
            key = getKey()
            
            if key == 'q':
                print("신호등 발견")
                speed= 0.05
                green_flag = 0
                
                startTimer()
                
            elif key == 'e':
                print("초록불이였다")
                green_flag = 1
                speed = 0.22

            elif key == 'z':
                print("시작하기")
                speed = 0.22
            elif key == 'm':
                break
            
            if current_speed != speed:
                print("속도" , speed)
                current_speed = speed
                twist.linear.x = speed
                pub.publish(twist)
    
    except rospy.ROSInterruptException: pass
    
    finally:
        twist = Twist()
        twist.linear.x = 0.0; twist.linear.y = 0.0; twist.linear.z = 0.0
        twist.angular.x = 0.0; twist.angular.y = 0.0; twist.angular.z = 0.0
        pub.publish(twist)
    