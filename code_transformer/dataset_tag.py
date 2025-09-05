# dataset_tagger.py (새 파일로 복사/수정 권장; 기존 Seq2Seq용 dataset.py는 보존)
import torch
from torch.utils.data import Dataset

TAGS = {"KEEP":0, "DEL":1, "REP":2, "APP":3}

class TrainTagDataset(Dataset):
    def __init__(self, pair_list, max_length, pad_id=0, bos_id=2, eos_id=3):
        """
        pair_list: [{"input_ids":[...], "output_ids":[...]}, ...]  (기존 전처리 재사용)
        """
        self.data = pair_list
        self.max_length = max_length
        self.pad_id, self.bos_id, self.eos_id = pad_id, bos_id, eos_id

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]["input_ids"][:self.max_length]
        y = self.data[idx]["output_ids"][:self.max_length]

        # 길이 보정
        L = max(len(x), len(y))
        if L < self.max_length:
            x = x + [self.pad_id]*(self.max_length - len(x))
            y = y + [self.pad_id]*(self.max_length - len(y))

        op_ids   = [TAGS["KEEP"]]*self.max_length
        rep_ids  = [self.pad_id]*self.max_length
        app_ids  = [self.pad_id]*self.max_length

        # 매우 단순한 정렬기반 라벨링(초기 버전): 1:1 위치 비교
        # 고급 구현: 최소편집거리로 정렬/정렬경로에서 DEL/REP/APP 산출 권장
        for i in range(self.max_length):
            xi, yi = x[i], y[i]
            if xi == self.pad_id and yi == self.pad_id:        # 둘 다 pad
                op_ids[i] = TAGS["KEEP"]
            elif xi in (self.bos_id, self.eos_id) or yi in (self.bos_id, self.eos_id):
                op_ids[i] = TAGS["KEEP"]
            elif yi == self.pad_id and xi != self.pad_id:       # 삭제 추정
                op_ids[i] = TAGS["DEL"]
            elif xi == yi:                                      # 보존
                op_ids[i] = TAGS["KEEP"]
            else:
                # 간단 규칙: 동일 위치가 다른 토큰이면 REPLACE
                op_ids[i]  = TAGS["REP"]
                rep_ids[i] = yi

        return {
            "input_ids": torch.tensor(x,  dtype=torch.long),
            "op_ids":    torch.tensor(op_ids, dtype=torch.long),
            "rep_ids":   torch.tensor(rep_ids, dtype=torch.long),
            "app_ids":   torch.tensor(app_ids, dtype=torch.long),  # 초기엔 미사용
        }
