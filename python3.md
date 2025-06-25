python3.md
Python에서 주의해야 할 점들
문법 관련 주의사항
들여쓰기 (Indentation)
Python은 들여쓰기로 코드 블록을 구분합니다
탭과 스페이스를 섞어 쓰면 안 됩니다
일관성 있게 스페이스 4개 또는 탭 사용

# 잘못된 예
# 올바른 예
if True:
print("Hello")  # 들여쓰기 없음 - 에러!
if True:
1234 print("Hello")  # 스페이스 4개, 4칸띄기기


대소문자 구분
Python은 대소문자를 구분합니다
Print와 print는 완전히 다른 것
![image](https://github.com/user-attachments/assets/3218ddd5-cf0d-427a-9f5b-756577673749)

# 잘못된 예
# 올바른 예
Print("Hello")  # 에러! (대문자 P)
print("Hello")  # 올바름





변수와 데이터 타입
변수명 규칙
숫자로 시작할 수 없음
특수문자 사용 불가 (밑줄 _ 제외)
예약어 사용 불가
# 잘못된 변수명
# 올바른 변수명
2name = "John"     # 숫자로 시작
my-name = "John"   # 하이픈 사용
class = "A"        # 예약어 사용
name2 = "John"
my_name = "John"
class_name = "A"


문자열 처리 주의사항
# 따옴표 주의
# 잘못된 경우
# 올바른 경우
text = "She said "Hello""  # 에러!
text = "She said \"Hello\""  # 올바름
text = 'She said "Hello"'   # 올바름



# 문자열과 숫자 연산
# 잘못된 경우
# 올바른 경우
age = 25
print("나이: " + age)   # 에러! 타입 불일치
age = 25
print("나이: " + str(age))   # 올바름
print(f"나이: {age}")        # 올바름 (f-string)









리스트와 인덱스
인덱스 범위 주의
my_list = [1, 2, 3]
print(my_list[3])  # 에러! 인덱스 범위 초과
print(my_list[2])  # 올바름 (마지막 요소)
print(my_list[-1]) # 올바름 (뒤에서 첫 번째)

리스트 복사 주의
list1 = [1, 2, 3]
list2 = list1        # 참조 복사 (같은 메모리)
list2.append(4)
print(list1)         # [1, 2, 3, 4] - 원본도 변경됨!

# 올바른 복사 방법
list2 = list1.copy()  # 또는 list1[:]

반복문과 조건문
무한 루프 주의
# 위험한 코드
while True:
    print("무한 루프!")  # Ctrl+C로 중단해야 함

# 안전한 코드
count = 0
while count < 10:
    print(f"카운트: {count}")
    count += 1  # 카운터 증가 잊지 말기!

조건문에서 할당 연산자 실수
x = 5
if x = 10:  # 에러! 할당 연산자 사용
    print("x는 10")

if x == 10:  # 올바름! 비교 연산자 사용
    print("x는 10")

자율주행관련 if문
자율주행 시스템 If문 예제 10개
🚗 센서 데이터 기반 판단 로직
1. 장애물 감지 및 긴급 제동
# 라이다 센서 데이터
front_distance = 2.5  # 미터
current_speed = 60    # km/h
brake_distance = (current_speed / 3.6) ** 2 / (2 * 7)  # 제동거리 계산

if front_distance <= brake_distance:
    emergency_brake = True
    brake_force = 100  # 최대 제동력
    print("긴급 제동 활성화")
elif front_distance <= brake_distance * 1.5:
    emergency_brake = False
    brake_force = 70
    print("강한 제동")
elif front_distance <= brake_distance * 2:
    emergency_brake = False
    brake_force = 40
    print("완만한 제동")
else:
    emergency_brake = False
    brake_force = 0
    print("정상 주행")

2. 차선 변경 가능성 판단
# 카메라 및 레이더 센서 데이터
left_lane_clear = True
left_rear_distance = 25.0  # 미터
left_front_distance = 30.0
left_vehicle_speed = 65    # km/h
my_speed = 70

if not left_lane_clear:
    lane_change_allowed = False
    action = "차선 변경 불가 - 차선 점유"
elif left_rear_distance < 20:
    lane_change_allowed = False
    action = "차선 변경 불가 - 후방 차량 근접"
elif left_front_distance < 25:
    lane_change_allowed = False
    action = "차선 변경 불가 - 전방 차량 근접"
elif abs(left_vehicle_speed - my_speed) > 20:
    lane_change_allowed = False
    action = "차선 변경 불가 - 속도 차이 과대"
else:
    lane_change_allowed = True
    action = "차선 변경 가능"
    
print(f"판단: {action}")

3. 신호등 인식 및 대응
# 컴퓨터 비전 처리 결과
traffic_light_color = "yellow"
distance_to_stop_line = 15.0  # 미터
current_speed = 50           # km/h
yellow_light_duration = 3    # 초

# 현재 속도로 정지선까지 도달 시간
time_to_stop_line = distance_to_stop_line / (current_speed / 3.6)

if traffic_light_color == "red":
    action = "정지"
    target_speed = 0
elif traffic_light_color == "green":
    action = "통과"
    target_speed = current_speed
elif traffic_light_color == "yellow":
    if time_to_stop_line <= yellow_light_duration and distance_to_stop_line > 5:
        action = "안전하게 통과"
        target_speed = current_speed
    elif distance_to_stop_line <= 5:
        action = "통과 (정지선 근접)"
        target_speed = current_speed
    else:
        action = "감속 후 정지"
        target_speed = 0
else:
    action = "신호등 미인식 - 서행"
    target_speed = 20

print(f"신호등: {traffic_light_color}, 행동: {action}, 목표속도: {target_speed}km/h")

4. 보행자 감지 및 회피
# 보행자 감지 시스템
pedestrian_detected = True
pedestrian_distance = 8.0      # 미터
pedestrian_speed = 1.5         # m/s (보행속도)
pedestrian_direction = "crossing"  # crossing, parallel, stationary
vehicle_speed = 40             # km/h

# 충돌 예상 시간 계산
vehicle_speed_ms = vehicle_speed / 3.6
time_to_collision = pedestrian_distance / vehicle_speed_ms

if not pedestrian_detected:
    action = "정상 주행"
    brake_intensity = 0
elif pedestrian_direction == "stationary" and pedestrian_distance > 5:
    action = "주의 주행"
    brake_intensity = 20
elif pedestrian_direction == "crossing":
    if time_to_collision <= 2:
        action = "긴급 제동"
        brake_intensity = 100
    elif time_to_collision <= 4:
        action = "강한 제동"
        brake_intensity = 80
    else:
        action = "예방 제동"
        brake_intensity = 40
elif pedestrian_direction == "parallel":
    action = "측면 주의"
    brake_intensity = 10
else:
    action = "보행자 행동 분석 중"
    brake_intensity = 30

print(f"보행자 상황: {action}, 제동강도: {brake_intensity}%")

5. 날씨 조건에 따른 주행 모드 조정
# 날씨 센서 및 도로 상태
weather_condition = "rain"
visibility = 150        # 미터
road_friction = 0.4     # 노면 마�찰계수 (0.0-1.0)
temperature = 2         # 섭씨
wind_speed = 15         # m/s

if weather_condition == "clear" and visibility > 500:
    driving_mode = "normal"
    max_speed_limit = 100
    following_distance_multiplier = 1.0
elif weather_condition == "rain":
    if road_friction < 0.3:
        driving_mode = "wet_cautious"
        max_speed_limit = 60
        following_distance_multiplier = 2.0
    else:
        driving_mode = "wet_normal"
        max_speed_limit = 80
        following_distance_multiplier = 1.5
elif weather_condition == "snow" or temperature <= 0:
    driving_mode = "winter"
    max_speed_limit = 50
    following_distance_multiplier = 2.5
elif weather_condition == "fog" or visibility < 100:
    driving_mode = "low_visibility"
    max_speed_limit = 40
    following_distance_multiplier = 2.0
elif wind_speed > 20:
    driving_mode = "high_wind"
    max_speed_limit = 70
    following_distance_multiplier = 1.3
else:
    driving_mode = "cautious"
    max_speed_limit = 80
    following_distance_multiplier = 1.2

print(f"주행모드: {driving_mode}, 제한속도: {max_speed_limit}km/h")
print(f"차간거리 배수: {following_distance_multiplier}배")

6. 주차 공간 감지 및 주차 가능성 판단
# 초음파 센서 및 카메라 데이터
parking_space_length = 5.2  # 미터
parking_space_width = 2.1   # 미터
vehicle_length = 4.5        # 미터
vehicle_width = 1.8         # 미터
obstacles_detected = False
space_angle = 5            # 주차공간 각도 (도)

# 주차 가능성 여유 공간 계산
length_margin = parking_space_length - vehicle_length
width_margin = parking_space_width - vehicle_width

if obstacles_detected:
    parking_possible = False
    parking_method = "주차 불가 - 장애물 감지"
elif length_margin < 0.5:
    parking_possible = False
    parking_method = "주차 불가 - 길이 부족"
elif width_margin < 0.2:
    parking_possible = False
    parking_method = "주차 불가 - 폭 부족"
elif space_angle > 15:
    parking_possible = False
    parking_method = "주차 불가 - 각도 부적절"
elif length_margin >= 1.0 and width_margin >= 0.5:
    parking_possible = True
    parking_method = "평행주차 - 여유공간 충분"
elif length_margin >= 0.7:
    parking_possible = True
    parking_method = "평행주차 - 정밀 조작 필요"
else:
    parking_possible = True
    parking_method = "평행주차 - 최소 공간"

print(f"주차 가능: {parking_possible}")
print(f"주차 방법: {parking_method}")
print(f"여유공간 - 길이: {length_margin:.1f}m, 폭: {width_margin:.1f}m")

7. 고속도로 합류 판단
# 고속도로 합류 상황 센서 데이터
main_lane_traffic_speed = 90    # km/h
merge_lane_length = 200         # 미터
current_position = 50           # 합류로에서의 현재 위치 (미터)
current_speed = 60             # km/h
gap_to_rear_vehicle = 80       # 미터
gap_to_front_vehicle = 120     # 미터
rear_vehicle_speed = 85        # km/h
front_vehicle_speed = 95       # km/h

# 합류 가능한 거리 계산
remaining_merge_distance = merge_lane_length - current_position
time_to_merge_end = remaining_merge_distance / (current_speed / 3.6)

if gap_to_rear_vehicle < 50 and gap_to_front_vehicle < 80:
    merge_action = "합류 대기 - 간격 부족"
    target_speed = max(40, current_speed - 10)
elif remaining_merge_distance < 50:
    if gap_to_rear_vehicle >= 30:
        merge_action = "즉시 합류 - 거리 부족"
        target_speed = main_lane_traffic_speed
    else:
        merge_action = "강제 감속 - 긴급상황"
        target_speed = 30
elif current_speed < main_lane_traffic_speed - 20:
    merge_action = "가속 후 합류"
    target_speed = min(main_lane_traffic_speed, current_speed + 20)
elif abs(current_speed - main_lane_traffic_speed) <= 10:
    merge_action = "적절한 타이밍에 합류"
    target_speed = main_lane_traffic_speed
else:
    merge_action = "속도 조정 후 합류"
    target_speed = main_lane_traffic_speed

print(f"합류 판단: {merge_action}")
print(f"목표 속도: {target_speed}km/h")
print(f"남은 합류 거리: {remaining_merge_distance}m")

8. 교차로 좌회전 안전성 판단
# 교차로 좌회전 상황
oncoming_vehicle_distance = 45   # 미터
oncoming_vehicle_speed = 55     # km/h
intersection_width = 20         # 미터
turn_completion_time = 4        # 좌회전 완료 예상 시간 (초)
yellow_light_remaining = 2      # 노란불 남은 시간 (초)
pedestrian_crossing = False     # 횡단보도 보행자 여부

↑ 직진차량 (대향차량)
    |
    |
←---+---→ (여기서 좌회전하려는 우리 차량)
    |
    ↓


# 대향 차량 도달 시간 계산
oncoming_arrival_time = oncoming_vehicle_distance / (oncoming_vehicle_speed / 3.6)

if pedestrian_crossing:
    turn_decision = "좌회전 대기 - 보행자 우선"
    action = "정지"
elif yellow_light_remaining > 0 and yellow_light_remaining < turn_completion_time:
    turn_decision = "좌회전 불가 - 신호 부족"
    action = "정지"
elif oncoming_arrival_time <= turn_completion_time + 1:
    turn_decision = "좌회전 대기 - 대향차량 근접"
    action = "대기"
elif oncoming_arrival_time <= turn_completion_time + 3:
    if oncoming_vehicle_speed > 60:
        turn_decision = "좌회전 대기 - 대향차량 고속"
        action = "대기"
    else:
        turn_decision = "신속한 좌회전 가능"
        action = "좌회전"
else:
    turn_decision = "안전한 좌회전 가능"
    action = "좌회전"

print(f"좌회전 판단: {turn_decision}")
print(f"행동: {action}")
print(f"대향차량 도달시간: {oncoming_arrival_time:.1f}초")

9. 차량 오작동 감지 및 안전 모드 전환
# 차량 시스템 상태 모니터링
steering_response = 0.8      # 조향 응답성 (0.0-1.0)
brake_system_pressure = 85   # 브레이크 압력 (%)
engine_temperature = 105     # 엔진 온도 (섭씨)
battery_voltage = 11.5       # 배터리 전압 (V)
tire_pressure_front = 1.8    # 전륜 타이어 압력 (bar)
tire_pressure_rear = 1.9     # 후륜 타이어 압력 (bar)
abs_system_active = True     # ABS 시스템 상태

if steering_response < 0.5:
    safety_mode = "긴급 정지"
    max_speed = 0
    warning_level = "위험"
elif brake_system_pressure < 60:
    safety_mode = "제동 보조"
    max_speed = 30
    warning_level = "위험"
elif engine_temperature > 120:
    safety_mode = "엔진 보호"
    max_speed = 40
    warning_level = "경고"
elif battery_voltage < 11.0:
    safety_mode = "전력 절약"
    max_speed = 50
    warning_level = "주의"
elif tire_pressure_front < 1.5 or tire_pressure_rear < 1.5:
    safety_mode = "타이어 주의"
    max_speed = 60
    warning_level = "주의"
elif not abs_system_active:
    safety_mode = "ABS 비활성"
    max_speed = 70
    warning_level = "주의"
else:
    safety_mode = "정상"
    max_speed = 100
    warning_level = "정상"

print(f"안전 모드: {safety_mode}")
print(f"최대 허용 속도: {max_speed}km/h")
print(f"경고 수준: {warning_level}")

10. 스쿨존 및 특수 구역 감지 대응
# GPS 및 도로 표지판 인식 데이터
current_zone = "school_zone"     # school_zone, hospital_zone, construction, normal
zone_speed_limit = 30           # km/h
time_of_day = 8                 # 시간 (0-23)
day_of_week = "monday"          # 요일
children_detected = True        # 어린이 감지 여부
construction_workers = False    # 공사 인부 감지 여부
current_speed = 45             # km/h

if current_zone == "school_zone":
    if 7 <= time_of_day <= 9 or 14 <= time_of_day <= 16:
        if day_of_week in ["monday", "tuesday", "wednesday", "thursday", "friday"]:
            enforced_speed_limit = 30
            extra_caution = True
        else:
            enforced_speed_limit = 40
            extra_caution = False
    else:
        enforced_speed_limit = 50
        extra_caution = False
    
    if children_detected:
        enforced_speed_limit = min(enforced_speed_limit, 25)
        extra_caution = True
        
elif current_zone == "hospital_zone":
    enforced_speed_limit = 30
    extra_caution = True
elif current_zone == "construction":
    if construction_workers:
        enforced_speed_limit = 20
        extra_caution = True
    else:
        enforced_speed_limit = 40
        extra_caution = False
else:
    enforced_speed_limit = 60
    extra_caution = False

# 속도 조정 판단
if current_speed > enforced_speed_limit:
    speed_action = "감속 필요"
    target_speed = enforced_speed_limit
elif current_speed > enforced_speed_limit * 0.9:
    speed_action = "속도 유지"
    target_speed = current_speed
else:
    speed_action = "적정 속도"
    target_speed = current_speed

print(f"구역: {current_zone}")
print(f"제한속도: {enforced_speed_limit}km/h")
print(f"특별 주의: {extra_caution}")
print(f"속도 조치: {speed_action} (목표: {target_speed}km/h)")

# 추가 안전 조치
if extra_caution:
    print("추가 조치: 전방 주시 강화, 비상등 점멸 고려")


🔍 자율주행 If문의 특징
1. 실시간 센서 데이터 처리
라이다, 카메라, 레이더, GPS 등 다중 센서 정보 융합
거리, 속도, 각도 등 정밀한 수치 계산
2. 안전 우선 논리
불확실한 상황에서는 항상 보수적 판단
다중 안전장치 및 페일세이프 메커니즘
3. 상황별 세분화된 판단
날씨, 시간, 도로 조건 등 환경 요소 고려
법규 준수 및 교통 상황 적응
4. 예측 기반 의사결정
시간 계산을 통한 충돌 예방
다른 교통 참여자 행동 예측
이러한 if문들은 실제 자율주행 시스템의 핵심 의사결정 로직을 보여줍니다!



함수 관련
매개변수 기본값 주의
# 위험한 코드
def add_item(item, my_list=[]):
    my_list.append(item)
    return my_list

# 문제: 기본값이 공유됨
list1 = add_item("apple")
list2 = add_item("banana")
print(list2)  # ['apple', 'banana'] - 예상과 다름!

# 올바른 코드
def add_item(item, my_list=None):
    if my_list is None:
        my_list = []
    my_list.append(item)
    return my_list

예외 처리
예외 처리 습관화
# 위험한 코드
number = int(input("숫자 입력: "))  # 문자 입력 시 에러!

# 안전한 코드
try:
    number = int(input("숫자 입력: "))
    result = 10 / number
    print(f"결과: {result}")
except ValueError:
    print("숫자를 입력해주세요")
except ZeroDivisionError:
    print("0으로 나눌 수 없습니다")

파일 처리
파일 닫기 잊지 말기
# 위험한 코드
file = open("data.txt", "r")
data = file.read()
# file.close() 잊음!

# 안전한 코드
with open("data.txt", "r") as file:
    data = file.read()
# 자동으로 파일이 닫힘

성능 관련
문자열 연결 최적화
# 비효율적
result = ""
for i in range(1000):
    result += str(i)  # 매번 새로운 문자열 객체 생성

# 효율적
result = "".join(str(i) for i in range(1000))

일반적인 실수들
print문에서 괄호 빠뜨리기
# Python 2 스타일 (에러!)
print "Hello"

# Python 3 스타일 (올바름)
print("Hello")

들여쓰기 혼용
# 에러 발생하는 코드
if True:
    print("Hello")  # 스페이스 4개
	print("World")  # 탭 문자 - 에러!

전역변수 사용 주의
count = 0

def increment():
    global count  # global 키워드 필요
    count += 1

def increment_wrong():
    count += 1  # 에러! 지역변수로 인식

이런 점들을 주의하면서 코딩하면 Python을 더 안전하고 효율적으로 사용할 수 있어요!

