from fastapi import FastAPI, HTTPException, Depends, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from typing import List, Optional
import uvicorn
from datetime import datetime
import logging
import json

from models import Base, Patient, Assessment
from inference_engine import MamdaniFuzzyInferenceEngine
from KB import ClusterBKnowledgeBase
from database import SessionLocal, engine

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize knowledge base and inference engine
kb = ClusterBKnowledgeBase()
inference_engine = MamdaniFuzzyInferenceEngine(kb)

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/assessment/{patient_id}", response_class=HTMLResponse)
async def start_assessment(request: Request, patient_id: int, db: Session = Depends(get_db)):
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    # Get first question
    first_question_id = list(kb.questions.keys())[0]
    question = kb.get_question(first_question_id)
    
    return templates.TemplateResponse(
        "question.html",
        {
            "request": request,
            "patient": patient,
            "question_id": first_question_id,
            "question": question,
            "is_age_question": first_question_id == "aspd_q9"
        }
    )

@app.post("/answer/{patient_id}")
async def submit_answer(
    patient_id: int,
    question_id: str = Form(...),
    answer: str = Form(...),
    db: Session = Depends(get_db)
):
    logger.info(f"Received answer for patient {patient_id}, question {question_id}: {answer}")
    
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    # Get the current assessment or create a new one if none exists
    assessment = db.query(Assessment).filter(
        Assessment.patient_id == patient_id,
        Assessment.completed == False
    ).first()
    
    if not assessment:
        assessment = Assessment(
            patient_id=patient_id,
            answers={},
            completed=False
        )
        db.add(assessment)
        db.flush()  # Flush to get the ID without committing
        logger.info(f"Created new assessment with ID: {assessment.id}")
    else:
        logger.info(f"Using existing assessment ID: {assessment.id}")
    
    # Store the answer - ensure answers dict exists and is properly initialized
    if assessment.answers is None:
        assessment.answers = {}
    
    # Validate answer format for fuzzy system
    valid_answers = ["never", "rarely", "sometimes", "often", "always", "yes", "no"]
    normalized_answer = answer.strip().lower()
    if normalized_answer not in valid_answers:
        logger.warning(f"Invalid answer '{answer}' received, defaulting to 'sometimes'")
        normalized_answer = "sometimes"
    
    # Store the normalized answer
    assessment.answers[question_id] = normalized_answer
    
    # CRITICAL FIX: Mark the answers field as modified for SQLAlchemy to detect changes
    from sqlalchemy.orm.attributes import flag_modified
    flag_modified(assessment, 'answers')
    
    # Commit the changes
    db.commit()
    db.refresh(assessment)
    
    logger.info(f"Stored answer. Current answers: {assessment.answers}")
    logger.info(f"Total answers stored: {len(assessment.answers) if assessment.answers else 0}")
    
    # Get next question
    question_ids = list(kb.questions.keys())
    try:
        current_index = question_ids.index(question_id)
    except ValueError:
        logger.error(f"Question ID {question_id} not found in knowledge base")
        raise HTTPException(status_code=400, detail="Invalid question ID")
    
    if current_index + 1 < len(question_ids):
        next_question_id = question_ids[current_index + 1]
        next_question = kb.get_question(next_question_id)
        logger.info(f"Moving to next question: {next_question_id}")
        return {
            "is_last": False,
            "next_question_id": next_question_id,
            "next_question": next_question,
            "is_age_question": next_question_id == "aspd_q9"
        }
    else:
        logger.info("Assessment completed, calculating results using Mamdani fuzzy inference")
        
        # Ensure we have the latest answers from the database
        db.refresh(assessment)
        current_answers = assessment.answers or {}
        logger.info(f"Final answers for calculation: {current_answers}")
        logger.info(f"Total questions answered: {len(current_answers)}")
        
        # Validate we have answers before processing
        if not current_answers:
            logger.error("No answers found in assessment")
            raise HTTPException(status_code=500, detail="No answers found for assessment")
        
        # Calculate results using the Mamdani fuzzy inference engine
        try:
            raw_scores = inference_engine.calculate_disorder_probability(current_answers)
            logger.info(f"Raw scores from Mamdani inference engine: {raw_scores}")
        except Exception as e:
            logger.error(f"Error in Mamdani fuzzy inference: {e}")
            raise HTTPException(status_code=500, detail=f"Error calculating scores: {str(e)}")
        
        # Scores from the new engine should already be in percentage format (0-100)
        formatted_scores = {}
        for disorder, score in raw_scores.items():
            try:
                float_score = float(score) if score is not None else 0.0
                # Ensure score is within valid range and cap at 85%
                formatted_score = max(0.0, min(85.0, float_score))
                formatted_scores[disorder] = round(formatted_score, 1)
            except (ValueError, TypeError):
                logger.error(f"Error processing score for {disorder}: {score}")
                formatted_scores[disorder] = 0.0
        
        logger.info(f"Formatted scores for storage: {formatted_scores}")
        
        # Generate recommendations
        try:
            recommendations = inference_engine.generate_enhanced_recommendations(formatted_scores)
            logger.info(f"Generated {len(recommendations)} recommendations")
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            recommendations = []
        
        # Generate detailed report if the engine supports it
        detailed_report = None
        try:
            if hasattr(inference_engine, 'generate_detailed_assessment_report'):
                detailed_report = inference_engine.generate_detailed_assessment_report(formatted_scores, current_answers)
                logger.info("Generated detailed assessment report with fuzzy insights")
        except Exception as e:
            logger.error(f"Error generating detailed report: {e}")
        
        # Update assessment with results
        assessment.scores = formatted_scores
        assessment.recommendations = recommendations
        assessment.completed = True
        
        # Store detailed report if available
        if detailed_report:
            assessment.detailed_report = detailed_report
            flag_modified(assessment, 'detailed_report')
        
        # Mark the fields as modified
        flag_modified(assessment, 'scores')
        flag_modified(assessment, 'recommendations')
        
        db.commit()
        db.refresh(assessment)
        
        logger.info(f"Assessment completed and saved with Mamdani fuzzy scores: {assessment.scores}")
        
        return {"is_last": True}

@app.get("/results/{patient_id}", response_class=HTMLResponse)
async def view_results(request: Request, patient_id: int, db: Session = Depends(get_db)):
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    # Get completed assessments with scores
    assessments = db.query(Assessment).filter(
        Assessment.patient_id == patient_id,
        Assessment.completed == True,
        Assessment.scores != None
    ).order_by(Assessment.created_at.desc()).all()
    
    logger.info(f"Found {len(assessments)} completed assessments for patient {patient_id}")
    
    # Ensure scores are properly formatted for display
    for assessment in assessments:
        if assessment.scores:
            logger.info(f"Retrieved scores from database: {assessment.scores}")
            
            # Ensure all scores are properly formatted as percentages
            processed_scores = {}
            for disorder, score in assessment.scores.items():
                try:
                    float_score = float(score) if score is not None else 0.0
                    processed_scores[disorder] = round(max(0.0, min(85.0, float_score)), 1)
                except (ValueError, TypeError):
                    logger.error(f"Error processing score for display: {disorder}={score}")
                    processed_scores[disorder] = 0.0
            
            assessment.scores = processed_scores
            logger.info(f"Processed scores for display: {assessment.scores}")
    
    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "patient": patient,
            "assessments": assessments
        }
    )

@app.delete("/assessments/{patient_id}")
async def delete_assessments(patient_id: int, db: Session = Depends(get_db)):
    # Delete all assessments for the patient
    deleted_count = db.query(Assessment).filter(Assessment.patient_id == patient_id).delete()
    db.commit()
    logger.info(f"Deleted {deleted_count} assessments for patient {patient_id}")
    return {"message": f"Deleted {deleted_count} assessments successfully"}

@app.post("/patient")
async def create_patient(
    name: str = Form(...),
    email: str = Form(...),
    db: Session = Depends(get_db)
):
    # Check if patient already exists
    existing_patient = db.query(Patient).filter(Patient.email == email).first()
    if existing_patient:
        logger.info(f"Patient with email {email} already exists")
        return {"id": existing_patient.id, "name": existing_patient.name, "email": existing_patient.email}
    
    # Create new patient if email doesn't exist
    patient = Patient(name=name, email=email)
    db.add(patient)
    db.commit()
    db.refresh(patient)
    logger.info(f"Created new patient: {patient.name} ({patient.email})")
    return {"id": patient.id, "name": patient.name, "email": patient.email}

@app.get("/patient/email/{email}")
async def get_patient_by_email(email: str, db: Session = Depends(get_db)):
    patient = db.query(Patient).filter(Patient.email == email).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return {"id": patient.id, "name": patient.name, "email": patient.email}

# Debug endpoints
@app.get("/debug/scores/{patient_id}")
async def debug_scores(patient_id: int, db: Session = Depends(get_db)):
    """Debug endpoint to check how scores are stored and retrieved"""
    assessments = db.query(Assessment).filter(
        Assessment.patient_id == patient_id
    ).order_by(Assessment.created_at.desc()).all()
    
    debug_info = []
    for assessment in assessments:
        debug_info.append({
            "id": assessment.id,
            "created_at": assessment.created_at,
            "completed": assessment.completed,
            "raw_scores": assessment.scores,
            "answers": assessment.answers,
            "answer_count": len(assessment.answers) if assessment.answers else 0,
            "recommendations_count": len(assessment.recommendations) if assessment.recommendations else 0,
            "has_detailed_report": hasattr(assessment, 'detailed_report') and assessment.detailed_report is not None
        })
    
    return {"debug_info": debug_info}

@app.get("/debug/kb")
async def debug_knowledge_base():
    """Debug endpoint to inspect the knowledge base structure"""
    
    debug_info = {
        "total_questions": len(kb.questions),
        "question_ids": list(kb.questions.keys()),
        "question_mapping": {}
    }
    
    # Get question details from the knowledge base
    for q_id in kb.questions.keys():
        question = kb.get_question(q_id)
        if question:
            debug_info["question_mapping"][q_id] = {
                "text": question.get("text", "No text available"),
                "impacts": question.get("impacts", {})
            }
    
    return debug_info

@app.get("/debug/test-scoring")
async def debug_test_scoring():
    """Test scoring with sample answers"""
    
    # Create test answers - mix of responses
    test_answers = {}
    question_ids = list(kb.questions.keys())
    
    # Add some variety to test scoring
    for i, q_id in enumerate(question_ids):
        if q_id == "aspd_q9":
            test_answers[q_id] = "yes"
        elif i % 4 == 0:
            test_answers[q_id] = "always"
        elif i % 4 == 1:
            test_answers[q_id] = "often"
        elif i % 4 == 2:
            test_answers[q_id] = "sometimes"
        else:
            test_answers[q_id] = "rarely"
    
    logger.info(f"Testing Mamdani fuzzy inference with {len(test_answers)} answers")
    
    # Calculate scores
    try:
        scores = inference_engine.calculate_disorder_probability(test_answers)
        
        # Generate detailed report for testing
        detailed_report = None
        if hasattr(inference_engine, 'generate_detailed_assessment_report'):
            detailed_report = inference_engine.generate_detailed_assessment_report(scores, test_answers)
        
        return {
            "success": True,
            "total_questions": len(question_ids),
            "answered_questions": len(test_answers),
            "test_answers": test_answers,
            "calculated_scores": scores,
            "has_detailed_report": detailed_report is not None,
            "inference_method": "Complete Mamdani Fuzzy Inference"
        }
    except Exception as e:
        logger.error(f"Error in test scoring: {e}")
        return {
            "success": False,
            "error": str(e),
            "test_answers": test_answers
        }

@app.get("/debug/fuzzy-systems")
async def debug_fuzzy_systems():
    """Debug endpoint to inspect the fuzzy systems in the inference engine"""
    
    debug_info = {
        "inference_type": "Complete Mamdani Fuzzy Inference",
        "fuzzy_systems": {}
    }
    
    # Check what fuzzy systems are initialized
    if hasattr(inference_engine, 'answer_fuzzy_systems'):
        debug_info["fuzzy_systems"]["answer_systems"] = len(inference_engine.answer_fuzzy_systems)
    
    if hasattr(inference_engine, 'symptom_aggregation_systems'):
        debug_info["fuzzy_systems"]["symptom_aggregation_systems"] = len(inference_engine.symptom_aggregation_systems)
    
    if hasattr(inference_engine, 'disorder_fuzzy_systems'):
        debug_info["fuzzy_systems"]["disorder_systems"] = len(inference_engine.disorder_fuzzy_systems)
    
    if hasattr(inference_engine, 'comorbidity_fuzzy_system'):
        debug_info["fuzzy_systems"]["has_comorbidity_system"] = True
    
    if hasattr(inference_engine, 'risk_assessment_system'):
        debug_info["fuzzy_systems"]["has_risk_assessment_system"] = True
    
    return debug_info

@app.get("/debug/current-assessment/{patient_id}")
async def debug_current_assessment(patient_id: int, db: Session = Depends(get_db)):
    """Debug current incomplete assessment"""
    assessment = db.query(Assessment).filter(
        Assessment.patient_id == patient_id,
        Assessment.completed == False
    ).first()
    
    if not assessment:
        return {"message": "No current assessment found"}
    
    return {
        "assessment_id": assessment.id,
        "answers": assessment.answers,
        "answer_count": len(assessment.answers) if assessment.answers else 0,
        "completed": assessment.completed,
        "created_at": assessment.created_at
    }

@app.get("/question/{patient_id}/{question_identifier}")
async def get_question(
    patient_id: int,
    question_identifier: str,
    db: Session = Depends(get_db)
):
    """Get a specific question by either its number or ID."""
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    # Get the current assessment
    assessment = db.query(Assessment).filter(
        Assessment.patient_id == patient_id,
        Assessment.completed == False
    ).first()
    
    if not assessment:
        raise HTTPException(status_code=404, detail="No active assessment found")
    
    # Get all question IDs
    question_ids = list(kb.questions.keys())
    
    # Determine if the identifier is a number or an ID
    try:
        question_number = int(question_identifier)
        # If it's a number, validate and get the corresponding ID
        if question_number < 1 or question_number > len(question_ids):
            raise HTTPException(status_code=400, detail="Invalid question number")
        question_id = question_ids[question_number - 1]
    except ValueError:
        # If it's not a number, use it as an ID directly
        question_id = question_identifier
    
    # Get the question
    question = kb.get_question(question_id)
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")
    
    return {
        "success": True,
        "question_id": question_id,
        "question": question,
        "is_age_question": question_id == "aspd_q9"
    }

@app.get("/answered-questions/{patient_id}")
async def get_answered_questions(
    patient_id: int,
    db: Session = Depends(get_db)
):
    """Get the list of answered question numbers for the current assessment."""
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    # Get the current assessment
    assessment = db.query(Assessment).filter(
        Assessment.patient_id == patient_id,
        Assessment.completed == False
    ).first()
    
    if not assessment or not assessment.answers:
        return {"answered_questions": []}
    
    # Convert question IDs to numbers
    question_ids = list(kb.questions.keys())
    answered_numbers = []
    
    for question_id in assessment.answers:
        try:
            number = question_ids.index(question_id) + 1
            answered_numbers.append(number)
        except ValueError:
            continue
    
    return {"answered_questions": answered_numbers}

@app.get("/question-mapping/{patient_id}")
async def get_question_mapping(
    patient_id: int,
    db: Session = Depends(get_db)
):
    """Get the question mapping and answered questions for the current assessment."""
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    # Get the current assessment
    assessment = db.query(Assessment).filter(
        Assessment.patient_id == patient_id,
        Assessment.completed == False
    ).first()
    
    if not assessment:
        return {
            "question_mapping": {},
            "answered_questions": [],
            "current_position": 1
        }
    
    # Get all question IDs
    question_ids = list(kb.questions.keys())
    
    # Create mapping of positions to question IDs
    question_mapping = {i+1: qid for i, qid in enumerate(question_ids)}
    
    # Get answered question positions
    answered_positions = []
    for question_id in assessment.answers:
        try:
            position = question_ids.index(question_id) + 1
            answered_positions.append(position)
        except ValueError:
            continue
    
    # Get current position
    current_position = 1
    if assessment.answers:
        last_answered = max(answered_positions) if answered_positions else 0
        current_position = last_answered + 1
    
    return {
        "question_mapping": question_mapping,
        "answered_questions": answered_positions,
        "current_position": current_position
    }

def process_metrics_data(assessments):
    """Process assessment data for metrics dashboard visualization."""
    if not assessments or len(assessments) < 2:
        return None
    
    # Sort assessments by date
    sorted_assessments = sorted(assessments, key=lambda x: x.created_at)
    
    # Extract dates and scores
    dates = [assessment.created_at.strftime('%Y-%m-%d') for assessment in sorted_assessments]
    disorders = list(sorted_assessments[0].scores.keys())
    
    # Extract scores for each disorder
    scores = {}
    for disorder in disorders:
        scores[disorder] = [float(assessment.scores.get(disorder, 0)) for assessment in sorted_assessments]
    
    # Calculate metrics
    latest_assessment = sorted_assessments[-1]
    previous_assessment = sorted_assessments[-2]
    
    improved_count = 0
    worsened_count = 0
    stable_count = 0
    
    for disorder in disorders:
        current_score = float(latest_assessment.scores.get(disorder, 0))
        previous_score = float(previous_assessment.scores.get(disorder, 0))
        
        # Calculate percentage change
        if previous_score == 0:
            change_percent = 0
        else:
            change_percent = ((current_score - previous_score) / previous_score) * 100
        
        if abs(change_percent) < 5:  # Within 5% is considered stable
            stable_count += 1
        elif change_percent < 0:  # Negative change means improvement (lower score is better)
            improved_count += 1
        else:
            worsened_count += 1
    
    # Calculate averages and find most variable disorder
    averages = {}
    variances = {}
    for disorder in disorders:
        disorder_scores = scores[disorder]
        avg = sum(disorder_scores) / len(disorder_scores)
        averages[disorder] = avg
        
        # Calculate variance
        variance = sum((x - avg) ** 2 for x in disorder_scores) / len(disorder_scores)
        variances[disorder] = variance
    
    highest_avg_disorder = max(averages.items(), key=lambda x: x[1])[0]
    most_variable_disorder = max(variances.items(), key=lambda x: x[1])[0]
    
    # Calculate overall trend
    first_scores = [float(score) for score in sorted_assessments[0].scores.values()]
    last_scores = [float(score) for score in sorted_assessments[-1].scores.values()]
    
    first_avg = sum(first_scores) / len(first_scores) if first_scores else 0
    last_avg = sum(last_scores) / len(last_scores) if last_scores else 0
    
    if first_avg == 0:
        overall_change = 0
    else:
        overall_change = ((first_avg - last_avg) / first_avg) * 100
    
    if overall_change > 5:
        overall_trend = "Improving"
    elif overall_change < -5:
        overall_trend = "Worsening"
    else:
        overall_trend = "Stable"
    
    # Calculate rate of change
    days_between = (sorted_assessments[-1].created_at - sorted_assessments[0].created_at).days
    if days_between > 0:
        rate_of_change = f"{abs(overall_change / days_between):.1f}% per day"
    else:
        rate_of_change = "N/A"
    
    # Generate insights with fuzzy linguistic descriptions
    trend_analysis = "Positive" if overall_change > 0 else "Negative" if overall_change < 0 else "Neutral"
    trend_description = f"Overall {trend_analysis.lower()} trend with {abs(overall_change):.1f}% change"
    
    # Risk level based on latest assessment average (using fuzzy linguistic terms)
    latest_avg = sum(last_scores) / len(last_scores) if last_scores else 0
    if latest_avg < 30:
        risk_level = "Low"
    elif latest_avg < 50:
        risk_level = "Low-Moderate"
    elif latest_avg < 70:
        risk_level = "Moderate-High"
    else:
        risk_level = "High"
    
    risk_description = f"Current risk level is {risk_level.lower()} based on Mamdani fuzzy assessment"
    
    # Treatment progress based on overall change
    treatment_progress = "Significant" if abs(overall_change) > 20 else "Moderate" if abs(overall_change) > 10 else "Minimal"
    treatment_description = f"{treatment_progress} progress observed in treatment outcomes"
    
    # Next steps based on trend
    next_steps = "Continue Current Treatment" if overall_change > 0 else "Review Treatment Plan"
    next_steps_description = "Consider adjusting treatment approach based on fuzzy assessment trends"
    
    return {
        "dates": dates,
        "disorders": disorders,
        "scores": scores,
        "improved_count": improved_count,
        "worsened_count": worsened_count,
        "stable_count": stable_count,
        "highest_avg_disorder": highest_avg_disorder,
        "most_variable_disorder": most_variable_disorder,
        "overall_trend": overall_trend,
        "rate_of_change": rate_of_change,
        "time_period": f"{days_between} days",
        "latest_date": dates[-1],
        "trend_analysis": trend_analysis,
        "trend_description": trend_description,
        "risk_level": risk_level,
        "risk_description": risk_description,
        "treatment_progress": treatment_progress,
        "treatment_description": treatment_description,
        "next_steps": next_steps,
        "next_steps_description": next_steps_description
    }

@app.get("/metrics/{patient_id}", response_class=HTMLResponse)
async def view_metrics(request: Request, patient_id: int, db: Session = Depends(get_db)):
    """View patient metrics dashboard."""
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    # Get completed assessments with scores
    assessments = db.query(Assessment).filter(
        Assessment.patient_id == patient_id,
        Assessment.completed == True,
        Assessment.scores != None
    ).order_by(Assessment.created_at).all()
    
    # Process metrics data
    metrics_data = process_metrics_data(assessments)
    
    template_data = {
        "request": request,
        "patient": patient,
        "assessments": assessments
    }
    
    if metrics_data:
        template_data.update(metrics_data)
    else:
        # Ensure all required variables are defined even when no metrics data
        template_data.update({
            "dates": [],
            "disorders": [],
            "scores": {},
            "improved_count": 0,
            "worsened_count": 0,
            "stable_count": 0,
            "highest_avg_disorder": "",
            "most_variable_disorder": "",
            "overall_trend": "",
            "rate_of_change": "",
            "time_period": "",
            "latest_date": "",
            "trend_analysis": "",
            "trend_description": "",
            "risk_level": "",
            "risk_description": "",
            "treatment_progress": "",
            "treatment_description": "",
            "next_steps": "",
            "next_steps_description": ""
        })
    
    return templates.TemplateResponse(
        "metrics.html",
        template_data
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)