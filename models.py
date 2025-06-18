from sqlalchemy import Column, Integer, String, DateTime, JSON, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import json

Base = declarative_base()

class Patient(Base):
    __tablename__ = "patients"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    email = Column(String, unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    assessments = relationship("Assessment", back_populates="patient")

class Assessment(Base):
    __tablename__ = "assessments"
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"))
    answers = Column(JSON, default=dict)
    scores = Column(JSON, default=dict)  # Store raw scores
    recommendations = Column(JSON, default=dict)
    completed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    detailed_report = Column(JSON, default=dict)  

    # Relationship
    patient = relationship("Patient", back_populates="assessments")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.answers = kwargs.get('answers', {})
        self.scores = kwargs.get('scores', {})
        self.recommendations = kwargs.get('recommendations', {})
        self.completed = kwargs.get('completed', False)
    
    def to_dict(self):
        return {
            'id': self.id,
            'patient_id': self.patient_id,
            'answers': self.answers,
            'scores': self.scores,
            'recommendations': self.recommendations,
            'completed': self.completed,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }