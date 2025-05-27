import json

class SpellingDataset:
    def _load_data(self, data_path):
        """데이터를 로드하고 전처리하는 메서드"""
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # 데이터가 'data' 키를 가지고 있는지 확인
        if 'data' not in raw_data:
            raise ValueError(f"데이터 파일 {data_path}에 'data' 키가 없습니다.")
        
        data = []
        for item in raw_data['data']:
            if 'annotation' not in item:
                continue
            
            annotation = item['annotation']
            if 'err_sentence' not in annotation or 'cor_sentence' not in annotation:
                continue
            
            # 입력과 출력 텍스트 추출
            input_text = annotation['err_sentence']
            output_text = annotation['cor_sentence']
            
            # 토큰화
            input_ids = self.tokenizer.encode(input_text)
            output_ids = self.tokenizer.encode(output_text)
            
            # 최대 길이 제한
            if len(input_ids) > self.max_length or len(output_ids) > self.max_length:
                continue
            
            data.append({
                'input_ids': input_ids,
                'output_ids': output_ids
            })
        
        return data 