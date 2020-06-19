import os
import pandas as pd
import numpy as np
from pathlib import Path
from module.simulator import Simulator

simulator = Simulator()
submission_ini = pd.read_csv(
    os.path.join(Path(__file__).resolve().parent, "sample_submission.csv")
)
order_ini = pd.read_csv(
    os.path.join(Path(__file__).resolve().parent, "extended_order.csv")
)


class Genome:
    def __init__(
        self, score_ini, input_len, output_len_1, output_len_2, h1=50, h2=50, h3=50
    ):
        # 평가 점수 초기화
        self.score = score_ini

        # 히든레이어 노드 개수
        self.hidden_layer1 = h1
        self.hidden_layer2 = h2
        self.hidden_layer3 = h3

        # Event_LINE_A 신경망 가중치 생성
        self.w1_1 = np.random.randn(input_len, self.hidden_layer1)
        self.w2_1 = np.random.randn(self.hidden_layer1, self.hidden_layer2)
        self.w3_1 = np.random.randn(self.hidden_layer2, self.hidden_layer3)
        self.w4_1 = np.random.randn(self.hidden_layer3, output_len_1)
        # Event_LINE_B 신경망 가중치 생성
        self.w1_2 = np.random.randn(input_len, self.hidden_layer1)
        self.w2_2 = np.random.randn(self.hidden_layer1, self.hidden_layer2)
        self.w3_2 = np.random.randn(self.hidden_layer2, self.hidden_layer3)
        self.w4_2 = np.random.randn(self.hidden_layer3, output_len_1)

        # LINE_A 수량 신경망 가중치 생성
        self.w5_1 = np.random.randn(input_len, self.hidden_layer1)
        self.w6_1 = np.random.randn(self.hidden_layer1, self.hidden_layer2)
        self.w7_1 = np.random.randn(self.hidden_layer2, self.hidden_layer3)
        self.w8_1 = np.random.randn(self.hidden_layer3, output_len_2)
        # LINE_B 수량 신경망 가중치 생성
        self.w5_2 = np.random.randn(input_len, self.hidden_layer1)
        self.w6_2 = np.random.randn(self.hidden_layer1, self.hidden_layer2)
        self.w7_2 = np.random.randn(self.hidden_layer2, self.hidden_layer3)
        self.w8_2 = np.random.randn(self.hidden_layer3, output_len_2)

        # change 이벤트 넣기
        self.change_event = [
            "CHANGE_12",
            "CHANGE_13",
            "CHANGE_14",
            "CHANGE_21",
            "CHANGE_23",
            "CHANGE_24",
            "CHANGE_31",
            "CHANGE_32",
            "CHANGE_34",
            "CHANGE_41",
            "CHANGE_42",
            "CHANGE_43",
        ]

        change_time = [6, 13, 13, 6, 13, 13, 13, 13, 6, 13, 13, 6]

        # LINE_A_Event 종류
        self.mask_1 = np.zeros([18], np.bool)  # 가능한 이벤트 검사용 마스크
        self.event_map_1 = {
            0: "CHECK_1",
            1: "CHECK_2",
            2: "CHECK_3",
            3: "CHECK_4",
            4: "PROCESS",
            5: "STOP",
        }

        count = 6
        for change in self.change_event:
            self.event_map_1[count] = change
            count += 1

        self.change_time_dict_1 = dict(zip(self.change_event, change_time))
        self.reversed_event_map_1 = dict(
            zip(list(self.event_map_1.values()), list(self.event_map_1.keys()))
        )

        # LINE_B_Event 종류
        self.mask_2 = np.zeros([18], np.bool)  # 가능한 이벤트 검사용 마스크
        self.event_map_2 = {
            0: "CHECK_1",
            1: "CHECK_2",
            2: "CHECK_3",
            3: "CHECK_4",
            4: "PROCESS",
            5: "STOP",
        }

        count = 6
        for change in self.change_event:
            self.event_map_2[count] = change
            count += 1
        self.change_time_dict_2 = dict(
            zip(self.change_event, change_time)
        )  # key는 change, value는 시간
        self.reversed_event_map_2 = dict(
            zip(list(self.event_map_2.values()), list(self.event_map_2.keys()))
        )
        # key는 event, value는 event_map에서 key

        # LINE_A_EVENT_STATUS
        self.check_time_1 = (
            28  # 28시간 검사를 완료했는지 검사, CHECK Event시 -1, processtime_time >=98 이면 28
        )
        self.process_1 = 0  # 생산 가능 여부, 0 이면 28 시간 검사 필요
        self.process_mode_1 = 0  # 생산 물품 번호 1~4, stop시 0
        self.process_time_1 = 0  # 생산시간이 얼마나 지속되었는지 검사, PROCESS +1, CHANGE +1, 최대 140
        # change 시간 요소
        self.change_time_1 = -1
        self.change_1 = 0
        # stop 시간 요소
        self.stop_time_1 = 0
        self.check_stop_time_1 = 0

        # LINE_B_EVENT_STATUS
        self.check_time_2 = (
            28  # 28시간 검사를 완료했는지 검사, CHECK Event시 -1, processtime_time >=98 이면 28
        )
        self.process_2 = 0  # 생산 가능 여부, 0 이면 28 시간 검사 필요
        self.process_mode_2 = 0  # 생산 물품 번호 1~4, stop시 0
        self.process_time_2 = 0  # 생산시간이 얼마나 지속되었는지 검사, PROCESS +1, CHANGE +1, 최대 140
        self.change_time_2 = -1
        self.change_2 = 0
        self.stop_time_2 = 0
        self.check_stop_time_2 = 0

    def update_mask_1(self):
        self.mask_1[:] = False

        if self.process_1 == 0:  # 140시간 process 마치고 딱 옴, check 할 차례 or check중
            if self.check_time_1 == 28:  # 아직 check 안함
                self.mask_1[:4] = True
                self.mask_1[5] = True  # stop도 할수 있음 check는 안돼
            elif self.check_time_1 < 28:  # check를 진행중
                self.mask_1[self.process_mode_1] = True  # 하던거만 마저해라

        if self.process_1 == 1:  # process 진행중.. 여기서는 change를 고려해줘야지
            self.mask_1[4] = True
            if self.process_time_1 > 98:  # 98 시간 지났으면 check나 stop 할수도 있지
                self.mask_1[:6] = True

            # ???????????????????????????????????????
            if self.process_time_1 > 0:
                # change event status check
                idx = self.process_mode_1  # 지금 생산중인 물건
                if self.change_1 == 1:  # process 내에서 change 진행중
                    self.mask_1[:] = False
                    if self.change_time_1 > 0:
                        self.mask_1[
                            self.reversed_event_map_1[self.previous_out1]
                        ] = True  # 하던 change 마저 하기
                    elif self.change_time_1 == 0:
                        self.mask_1[4] = True
                        self.change_1 = 0  # change 상태 False

                elif self.change_1 == 0:  # process 중인데 change는 안하는중
                    if self.process_time_1 > 0:
                        for change_index in range(3 * idx + 6, 3 * idx + 9):
                            # 140 시간 내에 끝낼 수 있는 change만 True로
                            if (
                                140
                                > self.process_time_1
                                + self.change_time_dict_1[
                                    self.change_event[change_index - 6]
                                ]
                            ):  ## >98
                                self.mask_1[change_index] = True

        if self.process_1 == 2:  # stop중이다!
            if self.stop_time_1 >= 28:  # stop이 28시간 이상이면 check도 되고 stop유지도 됨
                self.mask_1[:4] = True
                self.mask_1[5] = True
            if self.stop_time_1 < 28:  # 아직 28시간 안됐으면 계속 stop해야돼
                self.mask_1[5] = True
            if self.stop_time_1 == 192:  # stop마지노선 끝나면
                self.mask_1[:4] = True  # check 해야돼

    ###########################LINE  B ################################################
    def update_mask_2(self):
        self.mask_2[:] = False

        if self.process_2 == 0:  # 140시간 process 마치고 딱 옴, check 할 차례 or check중
            if self.check_time_2 == 28:  # 아직 check 안함
                self.mask_2[:4] = True
                self.mask_2[5] = True  # stop도 할수 있음 check는 안돼
            elif self.check_time_2 < 28:  # check를 진행중
                self.mask_2[self.process_mode_2] = True  # 하던거만 마저해라

        if self.process_2 == 1:  # process 진행중.. 여기서는 change를 고려해줘야지
            self.mask_2[4] = True
            if self.process_time_2 > 98:  # 98 시간 지났으면 check나 stop 할수도 있지
                self.mask_2[:6] = True

            if self.process_time_2 > 0:
                # change event status check
                idx = self.process_mode_2  # 지금 생산중인 물건
                if self.change_2 == 1:  # process 내에서 change 진행중
                    self.mask_2[:] = False
                    if self.change_time_2 > 0:
                        self.mask_2[
                            self.reversed_event_map_2[self.previous_out3]
                        ] = True  # 하던 change 마저 하기
                    elif self.change_time_2 == 0:
                        self.mask_2[4] = True
                        self.change_2 = 0  # change 상태 False

                elif self.change_2 == 0:  # process 중인데 change는 안하는중
                    if self.process_time_2 > 0:
                        for change_index in range(3 * idx + 6, 3 * idx + 9):
                            # 140 시간 내에 끝낼 수 있는 change만 True로
                            if (
                                140
                                > self.process_time_2
                                + self.change_time_dict_2[
                                    self.change_event[change_index - 6]
                                ]
                            ):  ## >98
                                self.mask_2[change_index] = True

        if self.process_2 == 2:  # stop중이다!
            if self.stop_time_2 >= 28:  # stop이 28시간 이상이면 check도 되고 stop유지도 됨
                self.mask_2[:4] = True
                self.mask_2[5] = True
            if self.stop_time_2 < 28:  # 아직 28시간 안됐으면 계속 stop해야돼
                self.mask_2[5] = True
            if self.stop_time_2 == 192:  # stop마지노선 끝나면
                self.mask_2[:4] = True  # check 해야돼

    def forward(self, inputs):
        # LINE_A_Event 신경망
        net = np.matmul(inputs, self.w1_1)
        net = self.linear(net)
        net = np.matmul(net, self.w2_1)
        net = self.linear(net)
        net = np.matmul(net, self.w3_1)
        net = self.sigmoid(net)
        net = np.matmul(net, self.w4_1)
        net = self.softmax(net)
        net += 1
        net = net * self.mask_1
        out1 = self.event_map_1[np.argmax(net)]  # 다음 event가 뭔지 랜덤 결정

        # LINE_A 수량 신경망
        net = np.matmul(inputs, self.w5_1)
        net = self.linear(net)
        net = np.matmul(net, self.w6_1)
        net = self.linear(net)
        net = np.matmul(net, self.w7_1)
        net = self.sigmoid(net)
        net = np.matmul(net, self.w8_1)
        net = self.softmax(net)
        out2 = np.argmax(net)
        out2 /= 5  # 수량도 랜덤 결정
        out2 += 0.2

        # LINE_B_Event 신경망
        net = np.matmul(inputs, self.w1_2)
        net = self.linear(net)
        net = np.matmul(net, self.w2_2)
        net = self.linear(net)
        net = np.matmul(net, self.w3_2)
        net = self.sigmoid(net)
        net = np.matmul(net, self.w4_2)
        net = self.softmax(net)
        net += 1
        net = net * self.mask_2
        out3 = self.event_map_2[np.argmax(net)]

        # LINE_B 수량 신경망
        net = np.matmul(inputs, self.w5_2)
        net = self.linear(net)
        net = np.matmul(net, self.w6_2)
        net = self.linear(net)
        net = np.matmul(net, self.w7_2)
        net = self.sigmoid(net)
        net = np.matmul(net, self.w8_2)
        net = self.softmax(net)
        out4 = np.argmax(net)
        out4 /= 5
        out4 += 0.2  # process 할때 수량 0이 들어가는것을 방지

        return out1, out2, out3, out4

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def linear(self, x):
        return x

    def create_order(self, order):
        for i in range(30):
            order.loc[91 + i, :] = ["0000-00-00", 0, 0, 0, 0]
        return order

    def predict(self, order):
        order = self.create_order(order)
        self.submission = submission_ini
        self.submission.loc[:, "PRT_1":"PRT_4"] = 0

        for s in range(self.submission.shape[0]):  # s: 2184시간
            self.update_mask_1()
            self.update_mask_2()
            # print(f"stage {s}")
            # print(f"mask_1 : {self.mask_1}")
            # print(f"mask_2 : {self.mask_2}")
            inputs = np.array(
                order.loc[s // 24 : (s // 24 + 30), "BLK_1":"BLK_4"]
            ).reshape(-1)
            inputs = np.append(inputs, s % 24)
            out1, out2, out3, out4 = self.forward(inputs)

            ################# LINE_A EVENT STATUS UPDATE ##################
            ##### CHECK ######
            if out1.startswith("CHECK"):
                if (
                    self.process_1 == 1 or self.process_1 == 2
                ):  # 이제 체크 할건데 지금 process중인거면
                    self.process_1 = 0  # process 중단해라
                    self.check_time_1 = 28  # 이제 시작할거야
                self.check_time_1 -= 1
                self.process_mode_1 = int(out1[-1]) - 1  # check1이면 0, 2면 1..

                if self.check_time_1 == 0:  # 이제 체크 끝난거면
                    self.process_1 = 1  # 생산 가능하지
                    self.process_time_1 = 0  # 생산할 거니까 시간은 0으로 초기화

            ###### PROCESS ######
            elif out1 == "PROCESS":
                self.process_time_1 += 1  # 생산중이었응께 1시간 추가
                if self.process_time_1 == 140:  # 140 시간 됐으면 process 중지하고 check할 준비
                    self.process_1 = 0
                    self.check_time_1 = 28

            #######CHANGE 넣기#######
            elif out1.startswith("CHANGE"):
                if self.change_1 == 0:
                    self.process_1 = 1
                    self.change_1 = 1  # change 상태 True
                    self.change_time_1 = self.change_time_dict_1[
                        out1
                    ]  # change time 초기화

                self.change_time_1 -= 1  # change time 흐른다
                self.process_time_1 += 1

                if self.change_time_1 == 0:  # change 끝났으면
                    self.process_mode_1 = (
                        int(out1[-1]) - 1
                    )  # EX)change12였으면 process2 활성화
                    # self.check_time_1 = 28 # 없어도 되지 않나??? change다음 check 안나오는데

            elif out3 == "STOP":
                if self.process_1 != 2:  # 방금 stop된거지
                    self.stop_time_1 = 0
                self.stop_time_1 += 1
                self.process_1 = 2
                if self.stop_time_1 == 192:
                    self.check_time_1 = 28

            ################### LINE_B EVENT STATUS UPDATE##############
            #### check ####
            if out3.startswith("CHECK"):
                if self.process_2 == 1 or self.process_2 == 2:
                    self.process_2 = 0
                    self.check_time_2 = 28
                self.check_time_2 -= 1
                self.process_time_2 += 1
                self.process_mode_2 = int(out3[-1]) - 1
                if self.check_time_2 == 0:
                    self.process_2 = 1
                    self.process_time_2 = 0

            #### process ####
            elif out3 == "PROCESS":
                self.process_time_2 += 1
                if self.process_time_2 == 140:
                    self.process_2 = 0
                    self.check_time_2 = 28
            #####CHANGE 넣기#####
            ## LINE B
            elif out3.startswith("CHANGE"):
                if self.change_2 == 0:
                    self.process_2 = 1
                    self.change_2 = 1  # change 상태 True
                    self.change_time_2 = self.change_time_dict_2[
                        out3
                    ]  # change time 초기화

                self.change_time_2 -= 1  # change time 흐른다
                self.process_time_2 += 1
                if self.change_time_2 == 0:  # change 끝났으면
                    self.process_mode_2 = (
                        int(out3[-1]) - 1
                    )  # EX)change12였으면 process2 활성화
                    # self.check_time_1 = 28 # 없어도 되지 않나??? change다음 check 안나오는데

            elif out3 == "STOP":
                if self.process_2 != 2:
                    self.stop_time_2 = 0
                self.stop_time_2 += 1
                self.process_2 = 2
                if self.stop_time_2 == 192:
                    self.check_time_2 = 28

            ############## out1, out3를 위로 던져주려고,,현재 상태랄까 ##############
            self.previous_out1 = out1
            self.previous_out3 = out3

            self.submission.loc[s, "Event_A"] = out1
            if self.submission.loc[s, "Event_A"] == "PROCESS":  # process였으면 그 수량을 넣어주고
                self.submission.loc[s, "MOL_A"] = out2
            else:
                self.submission.loc[s, "MOL_A"] = 0  # process 아니면 다 0

            self.submission.loc[s, "Event_B"] = out3
            if self.submission.loc[s, "Event_B"] == "PROCESS":
                self.submission.loc[s, "MOL_B"] = out4
            else:
                self.submission.loc[s, "MOL_B"] = 0

        # LINE A, B 23일간 MOL = 0
        self.submission.loc[: 24 * 23, "MOL_A"] = 0
        self.submission.loc[: 24 * 23, "MOL_B"] = 0

        # 2184시간 다 지났으니까 변수 초기화
        self.check_time_1 = 28
        self.process_1 = 0
        self.process_mode_1 = 0
        self.process_time_1 = 0
        self.change_1 = 0
        self.change_time_1 -= 1
        self.stop_time_1 = 0
        self.check_stop_time_1 = 192

        self.check_time_2 = 28
        self.process_2 = 0
        self.process_mode_2 = 0
        self.process_time_2 = 0
        self.change_2 = 0
        self.change_time_2 -= 1
        self.check_stop_time_2 = 192
        self.stop_time_2 = 0

        return self.submission


def genome_score(genome):
    submission = genome.predict(order_ini)
    genome.submission = submission
    genome.score, _ = simulator.get_score(submission)
    return genome
