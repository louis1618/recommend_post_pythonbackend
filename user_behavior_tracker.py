# user_behavior_tracker.py
from datetime import datetime
from sqlalchemy.orm import Session
from database import UserInteraction
import logging
from typing import Optional # 이 부분을 추가해주세요.

logger = logging.getLogger(__name__)

class UserBehaviorTracker:
    def __init__(self):
        self.engagement_weights = {
            'view': 0.1,         # 단순히 본 것
            'short_view': 0.05,  # 짧게 본 것은 낮은 점수 (부정적 신호로 간주할 수도 있음)
            'long_view': 0.8,    # 오래 본 것은 긍정적
            'detail_view': 1.0,  # 상세 페이지 진입은 매우 긍정적
            'like': 3.0,         # 좋아요는 강력한 긍정 신호
            'share': 5.0,        # 공유는 가장 강력한 긍정 신호
            'comment': 4.0,      # 댓글도 강력한 긍정 신호
            # 기타 상호작용 타입 추가 가능: save, follow, click 등
        }
    
    async def track_interaction(self, user_id: str, post_id: str, 
                              interaction_type: str, duration: Optional[float], db: Session) -> UserInteraction:
        """사용자 상호작용 추적 및 저장"""
        base_score = self.engagement_weights.get(interaction_type, 0.1) # 기본 점수 조정
        
        # 시청 시간 기반 추가 점수 (long_view, detail_view의 경우)
        if interaction_type in ['long_view', 'detail_view'] and duration is not None:
            # 10초당 0.5점 추가, 최대 5점까지
            base_score += min(duration / 10.0 * 0.5, 5.0) 
        
        interaction = UserInteraction(
            user_id=user_id,
            post_id=post_id,
            interaction_type=interaction_type,
            engagement_score=base_score,
            duration=duration,
            timestamp=datetime.utcnow()
        )
        
        db.add(interaction)
        db.commit()
        db.refresh(interaction)
        
        logger.info(f"Tracked interaction: user_id={user_id}, post_id={post_id}, type={interaction_type}, score={base_score:.2f}")
        return interaction