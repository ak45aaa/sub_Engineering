import serial
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ---- 1. 아두이노 연결 ----
arduino = serial.Serial('COM7', 9600)  # 포트는 네 컴퓨터 환경에 맞게 수정
time.sleep(2)  # 아두이노 리셋 기다리기

# ---- 2. 강화학습 정책 네트워크 ----
class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 4),  # 4개의 행동 output
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)

policy = PolicyNet()
optimizer = optim.Adam(policy.parameters(), lr=0.001)

# ---- 3. 학습 루프 ----
n_episodes = 500

for episode in range(n_episodes):
    log_probs = []
    rewards = []

    done = False
    total_reward = 0

    while not done:
        # ---- 3.1 아두이노로부터 센서 데이터 읽기 ----
        if arduino.in_waiting > 0:
            try:
                line = arduino.readline().decode('utf-8').strip()
                if line.startswith("Accel:"):
                    parts = line.split('||')
                    accel_str = parts[0].split(':')[1]
                    gyro_str = parts[1].split(':')[1]

                    accel = [float(x) for x in accel_str.split(',')]
                    gyro = [float(x) for x in gyro_str.split(',')]

                    obs = np.array(accel + gyro, dtype=np.float32)

                    # ---- 3.2 obs로 action 선택 ----
                    obs_tensor = torch.FloatTensor(obs)
                    probs = policy(obs_tensor)
                    m = torch.distributions.Categorical(probs)
                    action = m.sample()
                    log_prob = m.log_prob(action)

                    # ---- 3.3 action에 따라 deltaX, deltaY 계산 ----
                    # action은 0~3 중 하나 (X축 왼/오, Y축 왼/오)
                    delta = 5  # 각도 변화량 (+5도 또는 -5도)

                    deltaX = 0
                    deltaY = 0

                    if action.item() == 0:  # X축 왼쪽으로
                        deltaX = -delta
                    elif action.item() == 1:  # X축 오른쪽으로
                        deltaX = delta
                    elif action.item() == 2:  # Y축 왼쪽으로
                        deltaY = -delta
                    elif action.item() == 3:  # Y축 오른쪽으로
                        deltaY = delta

                    # ---- 3.4 deltaX, deltaY를 아두이노로 보내기 ----
                    # 1바이트씩 보내야 하니까 +128 offset해서 unsigned로 전송
                    arduino.write(bytes([deltaX + 128]))
                    arduino.write(bytes([deltaY + 128]))

                    # ---- 3.5 reward 계산 ----
                    # 목표: accel_x, accel_y가 0에 가까울수록 좋음
                    reward = - (abs(obs[0]) + abs(obs[1]))

                    log_probs.append(log_prob)
                    rewards.append(reward)
                    total_reward += reward

                    # ---- 3.6 종료조건 ----
                    if abs(obs[0]) > 1.0 or abs(obs[1]) > 1.0:
                        done = True

            except Exception as e:
                print("에러 발생:", e)

    # ---- 4. 에피소드 끝나고 정책 업데이트 ----
    loss = -torch.stack(log_probs).sum() * sum(rewards)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Episode {episode+1}: Total Reward = {total_reward:.2f}")

# ---- 5. 학습 완료 후 모델 저장 ----
torch.save(policy.state_dict(), "servo_policy_model.pth")
print("학습 완료!")
