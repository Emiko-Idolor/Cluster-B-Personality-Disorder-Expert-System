from KB import ClusterBKnowledgeBase
from inference_engine import InferenceEngine

def debug_scoring(self, answers):
    """Debug version of calculate_disorder_probability with detailed logging."""
    print("=== SCORING DEBUG ===")
    scores = {disorder: 0.0 for disorder in self.kb.disorders}
    
    print(f"Total questions answered: {len(answers)}")
    print(f"Available disorders: {list(self.kb.disorders)}")
    
    # Phase 1: Calculate base scores with detailed logging
    print("\n--- Phase 1: Base Score Calculation ---")
    for question_id, answer in answers.items():
        if not answer:
            continue
            
        weight = self.answer_mapping.get(answer, 0.0)
        print(f"\nQuestion {question_id}: {answer} (weight: {weight})")
        
        # Check if question is in cross-disorder impacts
        if question_id in self.cross_disorder_impacts:
            impacts = self.cross_disorder_impacts[question_id]
            print(f"  Using cross-disorder impacts: {impacts}")
        else:
            # Fallback to original single-disorder impact
            question = self.kb.get_question(question_id)
            if question and "impacts" in question:
                impacts = question["impacts"]
                print(f"  Using KB impacts: {impacts}")
            else:
                print(f"  WARNING: No impacts found for {question_id}")
                continue
        
        # Apply impacts to relevant disorders
        for disorder, impact in impacts.items():
            contribution = impact * weight
            scores[disorder] += contribution
            print(f"    {disorder}: +{contribution:.2f} (total: {scores[disorder]:.2f})")
    
    print(f"\n--- After Phase 1 (Base Scores) ---")
    for disorder, score in scores.items():
        print(f"{disorder}: {score:.2f}")
    
    # Phase 2: Apply comorbidity adjustments
    print("\n--- Phase 2: Comorbidity Adjustments ---")
    adjusted_scores = self._apply_comorbidity_adjustments(scores, answers)
    
    for disorder in scores:
        if adjusted_scores[disorder] != scores[disorder]:
            print(f"{disorder}: {scores[disorder]:.2f} -> {adjusted_scores[disorder]:.2f}")
    
    # Phase 3: Convert to percentages and apply clinical thresholds
    print("\n--- Phase 3: Clinical Threshold Scaling ---")
    final_scores = {}
    for disorder in adjusted_scores:
        disorder_info = self.kb.get_disorder_criteria_count(disorder)
        print(f"\n{disorder}:")
        
        if disorder_info:
            total_criteria = disorder_info["total_criteria"]
            min_criteria = disorder_info["min_criteria"]
            criteria_met = adjusted_scores[disorder]
            
            print(f"  Total criteria: {total_criteria}")
            print(f"  Min criteria needed: {min_criteria}")
            print(f"  Criteria met (score): {criteria_met:.2f}")
            
            # Calculate raw percentage
            raw_percentage = (criteria_met / total_criteria) * 100
            print(f"  Raw percentage: {raw_percentage:.1f}%")
            
            # Apply clinical significance threshold
            threshold = min_criteria * 0.5  # 50% of minimum criteria
            print(f"  Threshold (50% of min): {threshold:.2f}")
            
            if criteria_met >= threshold:
                print(f"  Above threshold - applying scaling")
                if criteria_met >= min_criteria:
                    # Above threshold - linear scaling
                    threshold_factor = min(criteria_met / min_criteria, 1.8)
                    final_score = raw_percentage * threshold_factor * 0.6
                    print(f"  Full threshold met - factor: {threshold_factor:.2f}, scaling: 0.6")
                else:
                    # Below threshold but approaching - reduced scaling
                    threshold_factor = (criteria_met / min_criteria) * 0.8
                    final_score = raw_percentage * threshold_factor * 0.4
                    print(f"  Approaching threshold - factor: {threshold_factor:.2f}, scaling: 0.4")
            else:
                # Well below threshold - minimal score
                final_score = raw_percentage * 0.1
                print(f"  Below threshold - minimal scaling: 0.1")
            
            print(f"  Final score before cap: {final_score:.1f}%")
            final_scores[disorder] = min(final_score, 85.0)  # Cap at 85%
            print(f"  Final score after cap: {final_scores[disorder]:.1f}%")
        else:
            print(f"  No disorder info found")
            final_scores[disorder] = 0.0
    
    print(f"\n--- Final Scores ---")
    for disorder, score in final_scores.items():
        print(f"{disorder}: {score:.1f}%")
    
    return final_scores

def test_debug_always_answers():
    """Test what happens when someone answers 'Always' to everything."""
    kb = ClusterBKnowledgeBase()
    engine = InferenceEngine(kb)
    
    # Add the debug method to the engine
    engine.debug_scoring = debug_scoring.__get__(engine, InferenceEngine)
    
    # Create answers with "Always" for all questions
    all_always_answers = {q_id: "Always" for q_id in kb.questions.keys()}
    
    print("Testing with ALL 'Always' answers:")
    scores = engine.debug_scoring(all_always_answers)
    
    return scores

if __name__ == "__main__":
    test_debug_always_answers()