from KB import ClusterBKnowledgeBase
import numpy as np
import math
from itertools import combinations

class InferenceEngine:
    def __init__(self, knowledge_base):
        self.kb = knowledge_base
        self.answer_mapping = {
            "never": 0.0,
            "rarely": 0.25,
            "sometimes": 0.5,
            "often": 0.75,
            "always": 1.0,
            "yes": 1.0,
            "no": 0.0
        }
        
        # Define disorder-specific question weights and criteria
        self.disorder_criteria = {
            "Borderline": {
                "questions": ["bpd_q1", "bpd_q2", "bpd_q3", "bpd_q4", "bpd_q5", 
                             "bpd_q6", "bpd_q7", "bpd_q8", "bpd_q9"],
                "min_criteria": 5,  # Need 5 out of 9 for BPD diagnosis
                "total_criteria": 9
            },
            "Antisocial": {
                "questions": ["aspd_q1", "aspd_q2", "aspd_q3", "aspd_q4", "aspd_q5", 
                             "aspd_q6", "aspd_q7", "aspd_q8", "aspd_q9"],
                "min_criteria": 3,  # Need 3 out of 7 for ASPD (plus age requirement)
                "total_criteria": 8  # 7 criteria + age requirement
            },
            "Histrionic": {
                "questions": ["hpd_q1", "hpd_q2", "hpd_q3", "hpd_q4", "hpd_q5", 
                             "hpd_q6", "hpd_q7", "hpd_q8"],
                "min_criteria": 5,  # Need 5 out of 8 for HPD
                "total_criteria": 8
            },
            "Narcissistic": {
                "questions": ["npd_q1", "npd_q2", "npd_q3", "npd_q4", "npd_q5", 
                             "npd_q6", "npd_q7", "npd_q8", "npd_q9"],
                "min_criteria": 5,  # Need 5 out of 9 for NPD
                "total_criteria": 9
            }
        }
        
        # Define comorbidity patterns and their interaction effects
        self.comorbidity_patterns = {
            ("Borderline", "Narcissistic"): {
                "multiplier": 1.15,
                "shared_symptoms": ["identity_disturbance", "anger_issues", "unstable_relationships"]
            },
            ("Borderline", "Histrionic"): {
                "multiplier": 1.12,
                "shared_symptoms": ["attention_seeking", "mood_instability", "theatrical"]
            },
            ("Narcissistic", "Histrionic"): {
                "multiplier": 1.10,
                "shared_symptoms": ["attention_seeking", "grandiosity", "shallow_emotions"]
            },
            ("Borderline", "Antisocial"): {
                "multiplier": 1.08,
                "shared_symptoms": ["impulsivity", "anger_issues", "unstable_relationships"]
            },
            ("Narcissistic", "Antisocial"): {
                "multiplier": 1.12,
                "shared_symptoms": ["lacks_empathy", "interpersonally_exploitative", "grandiosity"]
            },
            ("Histrionic", "Antisocial"): {
                "multiplier": 1.05,
                "shared_symptoms": ["attention_seeking", "shallow_emotions"]
            }
        }
    
    def calculate_disorder_probability(self, answers):
        """Calculate probability scores for each disorder based on answers."""
        print(f"DEBUG: Processing answers: {answers}")
        
        scores = {}
        
        for disorder, criteria_info in self.disorder_criteria.items():
            criteria_met = 0
            total_possible_score = 0
            actual_score = 0
            
            print(f"\nDEBUG: Processing {disorder}")
            print(f"DEBUG: Questions for {disorder}: {criteria_info['questions']}")
            
            for question_id in criteria_info['questions']:
                if question_id in answers:
                    answer = answers[question_id]
                    print(f"DEBUG: Question {question_id}, Answer: {answer}")
                    
                    # Handle age requirement for ASPD
                    if question_id == "aspd_q9":
                        if answer.lower() == "yes":
                            criteria_met += 1
                            actual_score += 1.0
                            print(f"DEBUG: Age requirement met for ASPD")
                        else:
                            print(f"DEBUG: Age requirement NOT met for ASPD - zeroing score")
                            scores[disorder] = 0.0
                            break
                        total_possible_score += 1.0
                        continue
                    
                    # Convert answer to numerical score
                    numerical_score = self.answer_mapping.get(answer.lower(), 0.0)
                    print(f"DEBUG: Converted '{answer}' to score: {numerical_score}")
                    
                    # Count as criterion met if score >= 0.5 (sometimes or higher)
                    if numerical_score >= 0.5:
                        criteria_met += 1
                        print(f"DEBUG: Criterion considered MET (score >= 0.5)")
                    
                    actual_score += numerical_score
                    total_possible_score += 1.0
                    
                    print(f"DEBUG: Running totals - Criteria met: {criteria_met}, Actual score: {actual_score}, Total possible: {total_possible_score}")
                else:
                    print(f"DEBUG: Question {question_id} not answered")
                    total_possible_score += 1.0
            
            # Skip if we already zeroed out the score (ASPD age requirement)
            if disorder in scores and scores[disorder] == 0.0:
                print(f"DEBUG: {disorder} score already zeroed, skipping calculation")
                continue
            
            # Calculate probability based on criteria met and clinical thresholds
            min_criteria = criteria_info['min_criteria']
            total_criteria = criteria_info['total_criteria']
            
            print(f"DEBUG: {disorder} - Criteria met: {criteria_met}/{total_criteria}, Min required: {min_criteria}")
            
            if criteria_met == 0:
                probability = 0.0
            elif criteria_met >= min_criteria:
                # High probability if meets diagnostic criteria
                base_probability = (criteria_met / total_criteria) * 80  # Max 80% for meeting criteria
                # Add bonus for exceeding minimum criteria
                excess_criteria = criteria_met - min_criteria
                bonus = min(15, excess_criteria * 3)  # Up to 15% bonus
                probability = min(85.0, base_probability + bonus)
            elif criteria_met >= min_criteria * 0.6:  # 60% of minimum criteria
                # Moderate probability for subclinical presentation
                probability = (criteria_met / total_criteria) * 50
            else:
                # Low probability for minimal criteria met
                probability = (criteria_met / total_criteria) * 25
            
            scores[disorder] = round(probability, 1)
            print(f"DEBUG: Final probability for {disorder}: {probability}%")
        
        # Apply comorbidity adjustments
        scores = self._apply_comorbidity_adjustments(scores, answers)
        
        print(f"\nDEBUG: Final scores after comorbidity adjustments: {scores}")
        return scores
    
    def _apply_comorbidity_adjustments(self, base_scores, answers):
        """Apply comorbidity pattern adjustments to base scores."""
        adjusted_scores = base_scores.copy()
        
        # Only apply comorbidity adjustments if multiple disorders have significant scores
        significant_disorders = [d for d, s in base_scores.items() if s >= 20.0]
        
        if len(significant_disorders) >= 2:
            print(f"DEBUG: Applying comorbidity adjustments for: {significant_disorders}")
            
            for (disorder1, disorder2), pattern_info in self.comorbidity_patterns.items():
                if disorder1 in significant_disorders and disorder2 in significant_disorders:
                    score1 = base_scores[disorder1]
                    score2 = base_scores[disorder2]
                    
                    # Calculate shared symptom strength
                    shared_strength = self._calculate_shared_symptom_strength(
                        answers, pattern_info["shared_symptoms"], disorder1, disorder2
                    )
                    
                    # Apply multiplier based on shared symptom strength
                    multiplier = 1.0 + (pattern_info["multiplier"] - 1.0) * shared_strength
                    
                    # Apply boost
                    adjusted_scores[disorder1] = min(85.0, score1 * multiplier)
                    adjusted_scores[disorder2] = min(85.0, score2 * multiplier)
                    
                    print(f"DEBUG: Comorbidity boost applied to {disorder1} and {disorder2}")
                    print(f"DEBUG: Multiplier: {multiplier}, Shared strength: {shared_strength}")
        
        return adjusted_scores
    
    def _calculate_shared_symptom_strength(self, answers, shared_symptoms, disorder1, disorder2):
        """Calculate how strongly shared symptoms are endorsed between two disorders."""
        if not shared_symptoms:
            return 0.0
        
        # Map symptom names to question IDs
        symptom_to_questions = {
            "identity_disturbance": ["bpd_q3"],
            "anger_issues": ["bpd_q8", "aspd_q4"],
            "unstable_relationships": ["bpd_q2"],
            "attention_seeking": ["hpd_q1"],
            "mood_instability": ["bpd_q6"],
            "theatrical": ["hpd_q6"],
            "grandiosity": ["npd_q1"],
            "shallow_emotions": ["hpd_q3"],
            "impulsivity": ["bpd_q4", "aspd_q3"],
            "lacks_empathy": ["npd_q7", "aspd_q7"],
            "interpersonally_exploitative": ["npd_q6"]
        }
        
        shared_endorsement = 0.0
        symptom_count = 0
        
        for symptom in shared_symptoms:
            if symptom in symptom_to_questions:
                for question_id in symptom_to_questions[symptom]:
                    if question_id in answers:
                        weight = self.answer_mapping.get(answers[question_id].lower(), 0.0)
                        shared_endorsement += weight
                        symptom_count += 1
        
        return shared_endorsement / max(symptom_count, 1)
    
    def identify_comorbidity_patterns(self, scores, threshold=20.0):
        """Identify likely comorbid patterns based on scores."""
        comorbid_patterns = []
        
        # Find disorders above threshold
        significant_disorders = [d for d, s in scores.items() if s >= threshold]
        
        if len(significant_disorders) >= 2:
            # Check all combinations of significant disorders
            for combo in combinations(significant_disorders, 2):
                disorder1, disorder2 = sorted(combo)
                pattern_key = (disorder1, disorder2)
                
                if pattern_key in self.comorbidity_patterns:
                    pattern_info = self.comorbidity_patterns[pattern_key]
                    comorbid_patterns.append({
                        "disorders": [disorder1, disorder2],
                        "scores": [scores[disorder1], scores[disorder2]],
                        "pattern_info": pattern_info,
                        "clinical_significance": self._assess_comorbidity_significance(
                            scores[disorder1], scores[disorder2]
                        )
                    })
        
        # Sort by clinical significance
        comorbid_patterns.sort(key=lambda x: x["clinical_significance"], reverse=True)
        return comorbid_patterns
    
    def _assess_comorbidity_significance(self, score1, score2):
        """Assess the clinical significance of a comorbid pattern."""
        avg_score = (score1 + score2) / 2
        balance_factor = 1 - abs(score1 - score2) / 100
        return avg_score * balance_factor
    
    def generate_recommendations(self, scores):
        """Generate recommendations considering comorbidity patterns."""
        recommendations = []
        comorbid_patterns = self.identify_comorbidity_patterns(scores)
        
        # Sort disorders by score
        sorted_disorders = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        for disorder, score in sorted_disorders:
            if score >= 15:  # Threshold for including in recommendations
                # Get disorder info from knowledge base if available
                disorder_info = self.kb.get_disorder_info(disorder) if hasattr(self.kb, 'get_disorder_info') else None
                
                # Check if this disorder is part of any comorbid patterns
                related_patterns = [p for p in comorbid_patterns 
                                 if disorder in p["disorders"]]
                
                rec = {
                    "disorder": disorder,
                    "score": round(score, 1),
                    "description": disorder_info["description"] if disorder_info else f"{disorder} Personality Disorder traits",
                    "criteria": disorder_info["criteria"] if disorder_info else [],
                    "action_steps": self._get_action_steps(disorder, score, related_patterns),
                    "comorbid_with": []
                }
                
                # Add comorbidity information
                for pattern in related_patterns:
                    other_disorder = [d for d in pattern["disorders"] if d != disorder][0]
                    rec["comorbid_with"].append({
                        "disorder": other_disorder,
                        "score": round(scores[other_disorder], 1),
                        "significance": round(pattern["clinical_significance"], 1)
                    })
                
                recommendations.append(rec)
        
        return recommendations
    
    def _get_action_steps(self, disorder, score, comorbid_patterns):
        """Get action steps considering comorbidity patterns."""
        # Base disorder-specific steps
        disorder_specific_steps = {
            "Borderline": [
                "Consider Dialectical Behavior Therapy (DBT)",
                "Practice mindfulness and emotional regulation techniques",
                "Learn distress tolerance and interpersonal effectiveness skills"
            ],
            "Antisocial": [
                "Consider Cognitive Behavioral Therapy (CBT)",
                "Work on developing empathy and moral reasoning",
                "Learn impulse control and anger management techniques"
            ],
            "Histrionic": [
                "Consider Psychodynamic or Schema Therapy",
                "Work on developing authentic emotional expression",
                "Practice building genuine, deeper relationships"
            ],
            "Narcissistic": [
                "Consider Psychodynamic or Schema Therapy",
                "Work on developing genuine empathy and self-awareness",
                "Practice accepting feedback and managing criticism"
            ]
        }
        
        steps = disorder_specific_steps.get(disorder, []).copy()
        
        # Add comorbidity-specific recommendations
        if comorbid_patterns:
            steps.insert(0, "Consider comprehensive assessment for multiple personality disorder traits")
            steps.insert(1, "Seek specialized treatment that addresses comorbid presentations")
        
        # Add severity-based steps
        if score >= 60:
            severity_steps = [
                "Seek immediate professional evaluation from a mental health specialist",
                "Consider intensive therapy or partial hospitalization if symptoms are severe",
                "Develop comprehensive safety and crisis management plans"
            ]
            steps = severity_steps + steps
        elif score >= 40:
            severity_steps = [
                "Schedule consultation with a mental health professional experienced in personality disorders",
                "Begin regular therapy sessions with a qualified therapist",
                "Consider medication evaluation for symptom management"
            ]
            steps = severity_steps + steps
        else:
            severity_steps = [
                "Monitor symptoms and their impact on relationships and daily functioning",
                "Consider psychoeducation about personality disorder traits",
                "Practice healthy coping strategies and self-care"
            ]
            steps = severity_steps + steps
        
        return steps

# Test function
def test_scoring():
    """Test the scoring system with sample data."""
    from KB import ClusterBKnowledgeBase
    
    kb = ClusterBKnowledgeBase()
    engine = InferenceEngine(kb)
    
    # Test with moderate BPD symptoms
    test_answers = {
        "bpd_q1": "often",      # Abandonment fears
        "bpd_q2": "sometimes",  # Unstable relationships
        "bpd_q3": "often",      # Identity disturbance
        "bpd_q4": "always",     # Impulsivity
        "bpd_q5": "rarely",     # Suicidal behavior
        "bpd_q6": "often",      # Mood instability
        "bpd_q7": "sometimes",  # Emptiness
        "bpd_q8": "often",      # Anger issues
        "bpd_q9": "rarely",     # Paranoid ideation
        
        # Low NPD symptoms
        "npd_q1": "rarely",     # Grandiosity
        "npd_q2": "never",      # Fantasies
        "npd_q3": "never",      # Special/unique
        "npd_q4": "sometimes",  # Requires admiration
        "npd_q5": "rarely",     # Entitlement
        "npd_q6": "never",      # Exploitative
        "npd_q7": "rarely",     # Lacks empathy
        "npd_q8": "never",      # Envious
        "npd_q9": "never",      # Arrogant
    }
    
    scores = engine.calculate_disorder_probability(test_answers)
    print("\nTest Results:")
    for disorder, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        print(f"{disorder}: {score}%")

if __name__ == "__main__":
    test_scoring()