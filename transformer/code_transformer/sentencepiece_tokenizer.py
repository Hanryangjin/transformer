import json
import sentencepiece as spm
import os
import sys

class SentencePieceTokenizer:
    def __init__(self, train_data_path, vocab_size=16000, max_length=512):
        print(f"토크나이저 초기화 시작: {train_data_path}")
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.train_data_path = train_data_path
        
        # 토크나이저 초기화
        self.tokenizer = spm.SentencePieceProcessor()
        
        # 모델 파일 경로
        self.model_path = os.path.abspath('spm_model.model')
        self.vocab_path = os.path.abspath('spm_model.vocab')
        
        # 모델이 이미 존재하는지 확인
        if not os.path.exists(self.model_path):
            print("토크나이저 모델이 없습니다. 새로 학습을 시작합니다.")
            self._train_tokenizer()
        else:
            print(f"기존 토크나이저 모델을 로드합니다: {self.model_path}")
            self.tokenizer.load(self.model_path)
    
    def _train_tokenizer(self):
        """SentencePiece 토크나이저를 학습하는 함수"""
        print(f"토크나이저 학습 시작: {self.train_data_path}")
        
        # 임시 파일 생성 (절대 경로 사용)
        temp_train_file = os.path.abspath('temp_train.txt')
        print(f"임시 파일 경로: {temp_train_file}")
        
        try:
            # 학습 데이터에서 텍스트 추출
            self._extract_texts(self.train_data_path, temp_train_file)
            
            # 임시 파일이 비어있는지 확인
            if not os.path.exists(temp_train_file):
                raise ValueError(f"임시 파일이 생성되지 않았습니다: {temp_train_file}")
                
            if os.path.getsize(temp_train_file) == 0:
                raise ValueError(f"추출된 텍스트가 없습니다. 파일 경로: {self.train_data_path}")
            
            print("SentencePiece 모델 학습 시작...")
            # SentencePiece 모델 학습
            model_prefix = os.path.splitext(self.model_path)[0]
            print(f"모델 저장 경로: {model_prefix}")
            
            # 학습 파라미터 설정
            train_args = {
                'input': temp_train_file,
                'model_prefix': model_prefix,
                'vocab_size': self.vocab_size,
                'character_coverage': 0.9995,
                'model_type': 'bpe',
                'pad_id': 0,
                'unk_id': 1,
                'bos_id': 2,
                'eos_id': 3,
                'pad_piece': '<pad>',
                'unk_piece': '<unk>',
                'bos_piece': '<s>',
                'eos_piece': '</s>'
            }
            
            print("학습 파라미터:", train_args)
            spm.SentencePieceTrainer.train(**train_args)
            
            # 임시 파일 삭제
            os.remove(temp_train_file)
            print("임시 파일 삭제 완료")
            
            # 학습된 모델 로드
            print(f"모델 파일 로드 시도: {self.model_path}")
            self.tokenizer.load(self.model_path)
            print("토크나이저 모델 로드 완료")
            
            
        except Exception as e:
            print(f"토크나이저 학습 중 오류 발생: {str(e)}", file=sys.stderr)
            # 임시 파일이 존재하면 삭제
            if os.path.exists(temp_train_file):
                os.remove(temp_train_file)
            raise
    
    def _extract_texts(self, data_path, output_file):
        """JSON 파일에서 텍스트를 추출하여 파일로 저장하는 함수"""
        print(f"데이터 파일 로드 중: {data_path}")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {data_path}")
        
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            print(f"데이터 구조 확인: {list(raw_data.keys())}")
            
            texts = []
            for item in raw_data['data']:
                # 데이터 구조 디버깅
                if len(texts) == 0:
                    print(f"첫 번째 아이템 구조: {list(item.keys())}")
                
                if 'cor_sentence' in item:
                    texts.append(item['cor_sentence'])
            
            print(f"추출된 문장 수: {len(texts)}")
            
            if len(texts) == 0:
                raise ValueError("추출된 문장이 없습니다. 데이터 구조를 확인해주세요.")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for text in texts:
                    f.write(text + '\n')
            
            print(f"임시 파일 생성 완료: {output_file}")
            
        except Exception as e:
            print(f"텍스트 추출 중 오류 발생: {str(e)}", file=sys.stderr)
            raise
    
    def encode(self, text):
        """텍스트를 토큰 ID로 변환하는 함수"""
        return self.tokenizer.encode(text)
    
    def decode(self, ids):
        """토큰 ID를 텍스트로 변환하는 함수"""
        return self.tokenizer.decode(ids)
    
    def tokenize(self, text):
        """텍스트를 토큰으로 분리하는 함수"""
        return self.tokenizer.encode_as_pieces(text)
    
    def convert_tokens_to_ids(self, tokens):
        """토큰을 ID로 변환하는 함수"""
        return [self.tokenizer.piece_to_id(token) for token in tokens]
    
    def convert_ids_to_tokens(self, ids):
        """ID를 토큰으로 변환하는 함수"""
        return [self.tokenizer.id_to_piece(id) for id in ids]
    
    def get_piece_size(self):
        """어휘 크기(토큰 수)를 반환하는 함수"""
        # SentencePieceProcessor에는 piece_size() 메서드가 있습니다.
        return self.tokenizer.piece_size()

#transformer_path = "D:\\김태호\\대학원 과목\\[Coding]\\transformer"
#sp = SentencePieceTokenizer(os.path.join(transformer_path, 'TrainData/combined_train_dataset.json'), vocab_size=16000, max_length=256)