from KB import ClusterBKnowledgeBase
import numpy as np
import math
from itertools import combinations
from skfuzzy import control as ctrl
import skfuzzy as fuzz

class MamdaniFuzzyInferenceEngine:
    def __init__(self, knowledge_base):
        self.kb = knowledge_base

        # Define disorder-specific question weights and criteria
        self.disorder_criteria = {
            "Borderline": {
                "questions": ["bpd_q1", "bpd_q2", "bpd_q3", "bpd_q4", "bpd_q5", 
                                "bpd_q6", "bpd_q7", "bpd_q8", "bpd_q9"],
                "min_criteria": 5,
                "total_criteria": 9,
                "weights": {
                    "bpd_q1": 1.2, "bpd_q2": 1.1, "bpd_q3": 1.0, "bpd_q4": 1.1,
                    "bpd_q5": 1.3, "bpd_q6": 1.2, "bpd_q7": 1.0, "bpd_q8": 1.0,
                    "bpd_q9": 0.8
                }
            },
            "Antisocial": {
                "questions": ["aspd_q1", "aspd_q2", "aspd_q3", "aspd_q4", "aspd_q5", 
                                "aspd_q6", "aspd_q7", "aspd_q8", "aspd_q9"],
                "min_criteria": 3,
                "total_criteria": 8,
                "weights": {
                    "aspd_q1": 1.2, "aspd_q2": 1.1, "aspd_q3": 1.0, "aspd_q4": 1.1,
                    "aspd_q5": 1.2, "aspd_q6": 1.0, "aspd_q7": 1.3, "aspd_q8": 1.1,
                    "aspd_q9": 2.0
                }
            },
            "Histrionic": {
                "questions": ["hpd_q1", "hpd_q2", "hpd_q3", "hpd_q4", "hpd_q5", 
                                "hpd_q6", "hpd_q7", "hpd_q8"],
                "min_criteria": 5,
                "total_criteria": 8,
                "weights": {
                    "hpd_q1": 1.2, "hpd_q2": 1.0, "hpd_q3": 1.1, "hpd_q4": 1.0,
                    "hpd_q5": 1.0, "hpd_q6": 1.1, "hpd_q7": 1.0, "hpd_q8": 1.0
                }
            },
            "Narcissistic": {
                "questions": ["npd_q1", "npd_q2", "npd_q3", "npd_q4", "npd_q5", 
                                "npd_q6", "npd_q7", "npd_q8", "npd_q9"],
                "min_criteria": 5,
                "total_criteria": 9,
                "weights": {
                    "npd_q1": 1.2, "npd_q2": 1.0, "npd_q3": 1.1, "npd_q4": 1.1,
                    "npd_q5": 1.1, "npd_q6": 1.2, "npd_q7": 1.3, "npd_q8": 1.0,
                    "npd_q9": 1.0
                }
            }
        }
        
        # Initialize all fuzzy systems
        self.answer_fuzzy_systems = self._initialize_answer_fuzzy_systems()
        self.symptom_aggregation_systems = self._initialize_symptom_aggregation_systems()
        self.disorder_fuzzy_systems = self._initialize_disorder_fuzzy_systems()
        self.comorbidity_fuzzy_system = self._initialize_comorbidity_fuzzy_system()
        self.risk_assessment_system = self._initialize_risk_assessment_system()
            
    def _initialize_answer_fuzzy_systems(self):
        """Initialize fuzzy systems for converting text answers to fuzzy symptom intensities."""
        systems = {}
        
        # Create a fuzzy system for each question type
        for disorder_name, criteria_info in self.disorder_criteria.items():
            for question_id in criteria_info['questions']:
                # Antecedent: Answer type (represented as a continuous variable)
                answer_input = ctrl.Antecedent(np.arange(0, 5.01, 0.01), f'{question_id}_answer')
                
                # Define membership functions for answer types
                answer_input['never'] = fuzz.trimf(answer_input.universe, [0, 0, 1])
                answer_input['rarely'] = fuzz.trimf(answer_input.universe, [0, 1, 2])
                answer_input['sometimes'] = fuzz.trimf(answer_input.universe, [1, 2, 3])
                answer_input['often'] = fuzz.trimf(answer_input.universe, [2, 3, 4])
                answer_input['always'] = fuzz.trimf(answer_input.universe, [3, 4, 5])
                answer_input['no'] = fuzz.trimf(answer_input.universe, [0, 0, 1])
                answer_input['yes'] = fuzz.trimf(answer_input.universe, [3, 4, 5])
                
                # Consequent: Symptom intensity
                symptom_intensity = ctrl.Consequent(np.arange(0, 101, 1), f'{question_id}_intensity')
                
                # Define membership functions for symptom intensity
                symptom_intensity['absent'] = fuzz.trimf(symptom_intensity.universe, [0, 0, 20])
                symptom_intensity['mild'] = fuzz.trimf(symptom_intensity.universe, [10, 25, 40])
                symptom_intensity['moderate'] = fuzz.trimf(symptom_intensity.universe, [30, 50, 70])
                symptom_intensity['severe'] = fuzz.trimf(symptom_intensity.universe, [60, 75, 90])
                symptom_intensity['extreme'] = fuzz.trimf(symptom_intensity.universe, [80, 100, 100])
                
                # Define rules for answer to symptom intensity mapping
                rules = [
                    ctrl.Rule(answer_input['never'] | answer_input['no'], symptom_intensity['absent']),
                    ctrl.Rule(answer_input['rarely'], symptom_intensity['mild']),
                    ctrl.Rule(answer_input['sometimes'], symptom_intensity['moderate']),
                    ctrl.Rule(answer_input['often'], symptom_intensity['severe']),
                    ctrl.Rule(answer_input['always'] | answer_input['yes'], symptom_intensity['extreme']),
                ]
                
                # Create control system
                answer_ctrl = ctrl.ControlSystem(rules)
                answer_sim = ctrl.ControlSystemSimulation(answer_ctrl)
                
                systems[question_id] = {
                    'input': answer_input,
                    'output': symptom_intensity,
                    'simulator': answer_sim
                }
        
        return systems
    
    def _initialize_symptom_aggregation_systems(self):
        """Initialize fuzzy systems for aggregating multiple symptoms into criteria satisfaction."""
        systems = {}
        
        for disorder in ["Borderline", "Antisocial", "Histrionic", "Narcissistic"]:
            # Antecedents: Average symptom intensity and consistency
            avg_intensity = ctrl.Antecedent(np.arange(0, 101, 1), f'{disorder}_avg_intensity')
            consistency = ctrl.Antecedent(np.arange(0, 101, 1), f'{disorder}_consistency')
            criteria_count = ctrl.Antecedent(np.arange(0, 1.01, 0.01), f'{disorder}_criteria_ratio')
            
            # Define membership functions for average intensity
            avg_intensity['very_low'] = fuzz.trimf(avg_intensity.universe, [0, 0, 25])
            avg_intensity['low'] = fuzz.trimf(avg_intensity.universe, [15, 30, 45])
            avg_intensity['moderate'] = fuzz.trimf(avg_intensity.universe, [35, 50, 65])
            avg_intensity['high'] = fuzz.trimf(avg_intensity.universe, [55, 70, 85])
            avg_intensity['very_high'] = fuzz.trimf(avg_intensity.universe, [75, 100, 100])
            
            # Define membership functions for consistency
            consistency['inconsistent'] = fuzz.trimf(consistency.universe, [0, 0, 35])
            consistency['somewhat_consistent'] = fuzz.trimf(consistency.universe, [25, 50, 75])
            consistency['consistent'] = fuzz.trimf(consistency.universe, [65, 100, 100])
            
            # Define membership functions for criteria ratio
            criteria_count['few'] = fuzz.trimf(criteria_count.universe, [0, 0, 0.3])
            criteria_count['some'] = fuzz.trimf(criteria_count.universe, [0.2, 0.4, 0.6])
            criteria_count['many'] = fuzz.trimf(criteria_count.universe, [0.5, 0.7, 0.9])
            criteria_count['most'] = fuzz.trimf(criteria_count.universe, [0.8, 1.0, 1.0])
            
            # Consequent: Aggregated symptom severity
            aggregated_severity = ctrl.Consequent(np.arange(0, 101, 1), f'{disorder}_aggregated')
            
            aggregated_severity['minimal'] = fuzz.trimf(aggregated_severity.universe, [0, 0, 20])
            aggregated_severity['low'] = fuzz.trimf(aggregated_severity.universe, [10, 25, 40])
            aggregated_severity['moderate'] = fuzz.trimf(aggregated_severity.universe, [30, 50, 70])
            aggregated_severity['high'] = fuzz.trimf(aggregated_severity.universe, [60, 80, 90])
            aggregated_severity['extreme'] = fuzz.trimf(aggregated_severity.universe, [85, 100, 100])
            
            # Define aggregation rules
            rules = [
                # Low intensity rules
                ctrl.Rule(avg_intensity['very_low'], aggregated_severity['minimal']),
                ctrl.Rule(avg_intensity['low'] & criteria_count['few'], aggregated_severity['minimal']),
                ctrl.Rule(avg_intensity['low'] & criteria_count['some'], aggregated_severity['low']),
                
                # Moderate intensity rules
                ctrl.Rule(avg_intensity['moderate'] & consistency['inconsistent'], aggregated_severity['low']),
                ctrl.Rule(avg_intensity['moderate'] & consistency['somewhat_consistent'] & criteria_count['some'], 
                         aggregated_severity['moderate']),
                ctrl.Rule(avg_intensity['moderate'] & consistency['consistent'] & criteria_count['many'], 
                         aggregated_severity['high']),
                
                # High intensity rules
                ctrl.Rule(avg_intensity['high'] & criteria_count['few'], aggregated_severity['moderate']),
                ctrl.Rule(avg_intensity['high'] & criteria_count['many'] & consistency['somewhat_consistent'], 
                         aggregated_severity['high']),
                ctrl.Rule(avg_intensity['high'] & criteria_count['most'] & consistency['consistent'], 
                         aggregated_severity['extreme']),
                
                # Very high intensity rules
                ctrl.Rule(avg_intensity['very_high'] & criteria_count['many'], aggregated_severity['extreme']),
                ctrl.Rule(avg_intensity['very_high'] & criteria_count['most'], aggregated_severity['extreme']),
                
                # Consistency modulation
                ctrl.Rule(consistency['inconsistent'] & criteria_count['most'], aggregated_severity['moderate']),
                ctrl.Rule(consistency['consistent'] & avg_intensity['moderate'] & criteria_count['most'], 
                         aggregated_severity['high']),
            ]
            
            # Create control system
            agg_ctrl = ctrl.ControlSystem(rules)
            agg_sim = ctrl.ControlSystemSimulation(agg_ctrl)
            
            systems[disorder] = {
                'avg_intensity': avg_intensity,
                'consistency': consistency,
                'criteria_count': criteria_count,
                'aggregated_severity': aggregated_severity,
                'simulator': agg_sim
            }
        
        return systems
    
    def _initialize_disorder_fuzzy_systems(self):
        """Initialize fuzzy systems for final disorder probability calculation."""
        systems = {}
        
        for disorder in ["Borderline", "Antisocial", "Histrionic", "Narcissistic"]:
            # Antecedents
            aggregated_severity = ctrl.Antecedent(np.arange(0, 101, 1), f'{disorder}_final_severity')
            clinical_threshold = ctrl.Antecedent(np.arange(0, 1.01, 0.01), f'{disorder}_threshold_met')
            
            # Define membership functions for aggregated severity
            aggregated_severity['minimal'] = fuzz.trimf(aggregated_severity.universe, [0, 0, 20])
            aggregated_severity['low'] = fuzz.trimf(aggregated_severity.universe, [10, 25, 40])
            aggregated_severity['moderate'] = fuzz.trimf(aggregated_severity.universe, [30, 50, 70])
            aggregated_severity['high'] = fuzz.trimf(aggregated_severity.universe, [60, 80, 90])
            aggregated_severity['extreme'] = fuzz.trimf(aggregated_severity.universe, [85, 100, 100])
            
            # Define membership functions for clinical threshold
            clinical_threshold['not_met'] = fuzz.trimf(clinical_threshold.universe, [0, 0, 0.4])
            clinical_threshold['partially_met'] = fuzz.trimf(clinical_threshold.universe, [0.3, 0.5, 0.7])
            clinical_threshold['met'] = fuzz.trimf(clinical_threshold.universe, [0.6, 0.8, 1.0])
            clinical_threshold['exceeded'] = fuzz.trimf(clinical_threshold.universe, [0.8, 1.0, 1.0])
            
            # Consequent: Disorder probability
            disorder_probability = ctrl.Consequent(np.arange(0, 101, 1), f'{disorder}_probability')
            
            disorder_probability['very_low'] = fuzz.trimf(disorder_probability.universe, [0, 0, 15])
            disorder_probability['low'] = fuzz.trimf(disorder_probability.universe, [5, 20, 35])
            disorder_probability['moderate'] = fuzz.trimf(disorder_probability.universe, [25, 40, 55])
            disorder_probability['high'] = fuzz.trimf(disorder_probability.universe, [45, 65, 80])
            disorder_probability['very_high'] = fuzz.trimf(disorder_probability.universe, [70, 85, 100])
            
            # Define disorder probability rules
            rules = [
                # Threshold not met
                ctrl.Rule(clinical_threshold['not_met'], disorder_probability['very_low']),
                ctrl.Rule(aggregated_severity['minimal'], disorder_probability['very_low']),
                
                # Low severity combinations
                ctrl.Rule(aggregated_severity['low'] & clinical_threshold['partially_met'], 
                         disorder_probability['low']),
                ctrl.Rule(aggregated_severity['low'] & clinical_threshold['met'], 
                         disorder_probability['moderate']),
                
                # Moderate severity combinations
                ctrl.Rule(aggregated_severity['moderate'] & clinical_threshold['partially_met'], 
                         disorder_probability['moderate']),
                ctrl.Rule(aggregated_severity['moderate'] & clinical_threshold['met'], 
                         disorder_probability['high']),
                ctrl.Rule(aggregated_severity['moderate'] & clinical_threshold['exceeded'], 
                         disorder_probability['high']),
                
                # High severity combinations
                ctrl.Rule(aggregated_severity['high'] & clinical_threshold['met'], 
                         disorder_probability['high']),
                ctrl.Rule(aggregated_severity['high'] & clinical_threshold['exceeded'], 
                         disorder_probability['very_high']),
                
                # Extreme severity
                ctrl.Rule(aggregated_severity['extreme'] & clinical_threshold['met'], 
                         disorder_probability['very_high']),
                ctrl.Rule(aggregated_severity['extreme'], disorder_probability['very_high']),
            ]
            
            # Create control system
            disorder_ctrl = ctrl.ControlSystem(rules)
            disorder_sim = ctrl.ControlSystemSimulation(disorder_ctrl)
            
            systems[disorder] = {
                'aggregated_severity': aggregated_severity,
                'clinical_threshold': clinical_threshold,
                'disorder_probability': disorder_probability,
                'simulator': disorder_sim
            }
        
        return systems
    
    def _initialize_comorbidity_fuzzy_system(self):
        """Initialize fuzzy system for comorbidity adjustments."""
        # Antecedents for pairs of disorders
        disorder1_prob = ctrl.Antecedent(np.arange(0, 101, 1), 'disorder1_probability')
        disorder2_prob = ctrl.Antecedent(np.arange(0, 101, 1), 'disorder2_probability')
        interaction_type = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'interaction_strength')
        
        # Define membership functions for disorder probabilities
        for disorder_prob in [disorder1_prob, disorder2_prob]:
            disorder_prob['low'] = fuzz.trimf(disorder_prob.universe, [0, 0, 35])
            disorder_prob['moderate'] = fuzz.trimf(disorder_prob.universe, [25, 50, 75])
            disorder_prob['high'] = fuzz.trimf(disorder_prob.universe, [65, 100, 100])
        
        # Define membership functions for interaction strength
        interaction_type['weak'] = fuzz.trimf(interaction_type.universe, [0, 0, 0.4])
        interaction_type['moderate'] = fuzz.trimf(interaction_type.universe, [0.3, 0.5, 0.7])
        interaction_type['strong'] = fuzz.trimf(interaction_type.universe, [0.6, 1.0, 1.0])
        
        # Consequent: Comorbidity adjustment factor
        adjustment_factor = ctrl.Consequent(np.arange(0.8, 1.31, 0.01), 'adjustment_factor')
        
        adjustment_factor['none'] = fuzz.trimf(adjustment_factor.universe, [0.8, 0.9, 1.0])
        adjustment_factor['small'] = fuzz.trimf(adjustment_factor.universe, [0.95, 1.05, 1.15])
        adjustment_factor['moderate'] = fuzz.trimf(adjustment_factor.universe, [1.1, 1.15, 1.2])
        adjustment_factor['large'] = fuzz.trimf(adjustment_factor.universe, [1.15, 1.25, 1.3])
        
        # Define comorbidity rules
        rules = [
            # Low probability combinations
            ctrl.Rule(disorder1_prob['low'] | disorder2_prob['low'], adjustment_factor['none']),
            
            # Moderate probability combinations
            ctrl.Rule(disorder1_prob['moderate'] & disorder2_prob['moderate'] & interaction_type['weak'], 
                     adjustment_factor['small']),
            ctrl.Rule(disorder1_prob['moderate'] & disorder2_prob['moderate'] & interaction_type['moderate'], 
                     adjustment_factor['moderate']),
            ctrl.Rule(disorder1_prob['moderate'] & disorder2_prob['high'] & interaction_type['strong'], 
                     adjustment_factor['moderate']),
            
            # High probability combinations
            ctrl.Rule(disorder1_prob['high'] & disorder2_prob['high'] & interaction_type['weak'], 
                     adjustment_factor['moderate']),
            ctrl.Rule(disorder1_prob['high'] & disorder2_prob['high'] & interaction_type['moderate'], 
                     adjustment_factor['large']),
            ctrl.Rule(disorder1_prob['high'] & disorder2_prob['high'] & interaction_type['strong'], 
                     adjustment_factor['large']),
            
            # Mixed combinations
            ctrl.Rule(disorder1_prob['high'] & disorder2_prob['moderate'] & interaction_type['strong'], 
                     adjustment_factor['moderate']),
        ]
        
        # Create control system
        comorbidity_ctrl = ctrl.ControlSystem(rules)
        comorbidity_sim = ctrl.ControlSystemSimulation(comorbidity_ctrl)
        
        return {
            'disorder1_prob': disorder1_prob,
            'disorder2_prob': disorder2_prob,
            'interaction_type': interaction_type,
            'adjustment_factor': adjustment_factor,
            'simulator': comorbidity_sim
        }
    
    def _initialize_risk_assessment_system(self):
        """Initialize fuzzy system for risk assessment."""
        # Antecedents
        self_harm_risk = ctrl.Antecedent(np.arange(0, 101, 1), 'self_harm_risk')
        aggression_risk = ctrl.Antecedent(np.arange(0, 101, 1), 'aggression_risk')
        overall_severity = ctrl.Antecedent(np.arange(0, 101, 1), 'overall_severity')
        
        # Define membership functions
        for risk in [self_harm_risk, aggression_risk, overall_severity]:
            risk['low'] = fuzz.trimf(risk.universe, [0, 0, 35])
            risk['moderate'] = fuzz.trimf(risk.universe, [25, 50, 75])
            risk['high'] = fuzz.trimf(risk.universe, [65, 100, 100])
        
        # Consequent: Overall risk level
        risk_level = ctrl.Consequent(np.arange(0, 101, 1), 'risk_level')
        
        risk_level['low'] = fuzz.trimf(risk_level.universe, [0, 0, 30])
        risk_level['moderate'] = fuzz.trimf(risk_level.universe, [20, 40, 60])
        risk_level['high'] = fuzz.trimf(risk_level.universe, [50, 70, 85])
        risk_level['critical'] = fuzz.trimf(risk_level.universe, [75, 100, 100])
        
        # Define risk assessment rules
        rules = [
            # Low risk scenarios
            ctrl.Rule(self_harm_risk['low'] & aggression_risk['low'], risk_level['low']),
            ctrl.Rule(overall_severity['low'], risk_level['low']),
            
            # Moderate risk scenarios
            ctrl.Rule(self_harm_risk['moderate'] | aggression_risk['moderate'], risk_level['moderate']),
            ctrl.Rule(overall_severity['moderate'] & self_harm_risk['low'], risk_level['moderate']),
            
            # High risk scenarios
            ctrl.Rule(self_harm_risk['high'], risk_level['high']),
            ctrl.Rule(aggression_risk['high'], risk_level['high']),
            ctrl.Rule(overall_severity['high'] & (self_harm_risk['moderate'] | aggression_risk['moderate']), 
                     risk_level['high']),
            
            # Critical risk scenarios
            ctrl.Rule(self_harm_risk['high'] & aggression_risk['high'], risk_level['critical']),
            ctrl.Rule(self_harm_risk['high'] & overall_severity['high'], risk_level['critical']),
        ]
        
        # Create control system
        risk_ctrl = ctrl.ControlSystem(rules)
        risk_sim = ctrl.ControlSystemSimulation(risk_ctrl)
        
        return {
            'self_harm_risk': self_harm_risk,
            'aggression_risk': aggression_risk,
            'overall_severity': overall_severity,
            'risk_level': risk_level,
            'simulator': risk_sim
        }
    
    def _convert_answer_to_fuzzy_input(self, answer):
        """Convert text answer to fuzzy input value."""
        answer_map = {
            "never": 0.0,
            "rarely": 1.0,
            "sometimes": 2.0,
            "often": 3.0,
            "always": 4.0,
            "no": 0.0,
            "yes": 4.0
        }
        return answer_map.get(answer.lower(), 2.0)  # Default to 'sometimes' if unknown
    
    def _calculate_symptom_intensities(self, answers):
        """Calculate fuzzy symptom intensities from answers."""
        symptom_intensities = {}
        
        for question_id, answer in answers.items():
            if question_id in self.answer_fuzzy_systems:
                try:
                    system = self.answer_fuzzy_systems[question_id]
                    simulator = system['simulator']
                    
                    # Convert answer to fuzzy input
                    fuzzy_input = self._convert_answer_to_fuzzy_input(answer)
                    simulator.input[f'{question_id}_answer'] = fuzzy_input
                    
                    # Compute fuzzy output
                    simulator.compute()
                    
                    # Get defuzzified output
                    intensity = simulator.output[f'{question_id}_intensity']
                    symptom_intensities[question_id] = intensity
                    
                except Exception as e:
                    print(f"Error processing {question_id}: {e}")
                    symptom_intensities[question_id] = 0.0
        
        return symptom_intensities
    
    def _aggregate_symptoms_by_disorder(self, symptom_intensities, disorder):
        """Aggregate symptoms for a specific disorder using fuzzy logic."""
        criteria_info = self.disorder_criteria[disorder]
        disorder_symptoms = []
        weights = []
        
        # Collect relevant symptoms and weights
        for question_id in criteria_info['questions']:
            if question_id in symptom_intensities:
                disorder_symptoms.append(symptom_intensities[question_id])
                weights.append(criteria_info['weights'].get(question_id, 1.0))
        
        if not disorder_symptoms:
            return 0.0, 0.0, 0.0
        
        # Calculate weighted average intensity
        weighted_sum = sum(s * w for s, w in zip(disorder_symptoms, weights))
        total_weight = sum(weights)
        avg_intensity = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        # Calculate consistency (inverse of standard deviation)
        if len(disorder_symptoms) > 1:
            mean = np.mean(disorder_symptoms)
            std = np.std(disorder_symptoms)
            # Convert to 0-100 scale where 100 is perfectly consistent
            consistency = max(0, 100 * (1 - std / 50))  # Assuming max std of 50
        else:
            consistency = 50.0  # Neutral consistency for single symptom
        
        # Calculate criteria ratio
        high_symptoms = sum(1 for s in disorder_symptoms if s >= 50)  # Count moderate+ symptoms
        criteria_ratio = high_symptoms / len(disorder_symptoms) if disorder_symptoms else 0.0
        
        return avg_intensity, consistency, criteria_ratio
    
    def _apply_symptom_aggregation(self, disorder, avg_intensity, consistency, criteria_ratio):
        """Apply fuzzy symptom aggregation for a disorder."""
        try:
            system = self.symptom_aggregation_systems[disorder]
            simulator = system['simulator']
            
            # Set inputs
            simulator.input[f'{disorder}_avg_intensity'] = avg_intensity
            simulator.input[f'{disorder}_consistency'] = consistency
            simulator.input[f'{disorder}_criteria_ratio'] = criteria_ratio
            
            # Compute
            simulator.compute()
            
            # Get output
            aggregated_severity = simulator.output[f'{disorder}_aggregated']
            return aggregated_severity
            
        except Exception as e:
            print(f"Error in symptom aggregation for {disorder}: {e}")
            return avg_intensity  # Fallback to average intensity
    
    def _calculate_clinical_threshold_ratio(self, answers, disorder):
        """Calculate clinical threshold satisfaction using fuzzy logic."""
        criteria_info = self.disorder_criteria[disorder]
        symptom_intensities = self._calculate_symptom_intensities(answers)
        
        # Count symptoms meeting clinical threshold (intensity >= 50)
        met_count = 0
        for question_id in criteria_info['questions']:
            if question_id in symptom_intensities:
                if symptom_intensities[question_id] >= 50:  # Clinical threshold
                    met_count += 1
        
        # Calculate ratios
        actual_ratio = met_count / len(criteria_info['questions'])
        required_ratio = criteria_info['min_criteria'] / criteria_info['total_criteria']
        
        # Calculate how well the threshold is met (0-1 scale)
        if required_ratio > 0:
            threshold_satisfaction = min(1.0, actual_ratio / required_ratio)
        else:
            threshold_satisfaction = 1.0
        
        return threshold_satisfaction
    
    def _calculate_final_disorder_probability(self, disorder, aggregated_severity, threshold_ratio):
        """Calculate final disorder probability using fuzzy logic."""
        try:
            system = self.disorder_fuzzy_systems[disorder]
            simulator = system['simulator']
            
            # Set inputs
            simulator.input[f'{disorder}_final_severity'] = aggregated_severity
            simulator.input[f'{disorder}_threshold_met'] = threshold_ratio
            
            # Compute
            simulator.compute()
            
            # Get output
            probability = simulator.output[f'{disorder}_probability']
            return probability
            
        except Exception as e:
            print(f"Error calculating probability for {disorder}: {e}")
            return 0.0
    
    def _apply_comorbidity_adjustment(self, disorder1, disorder2, prob1, prob2, interaction_strength):
        """Apply fuzzy comorbidity adjustment."""
        try:
            simulator = self.comorbidity_fuzzy_system['simulator']
            
            # Set inputs
            simulator.input['disorder1_probability'] = prob1
            simulator.input['disorder2_probability'] = prob2
            simulator.input['interaction_strength'] = interaction_strength
            
            # Compute
            simulator.compute()
            
            # Get adjustment factor
            adjustment_factor = simulator.output['adjustment_factor']
            return adjustment_factor
            
        except Exception as e:
            print(f"Error in comorbidity adjustment: {e}")
            return 1.0  # No adjustment on error
    
    def calculate_disorder_probability(self, answers):
        """Calculate probability scores using complete Mamdani fuzzy inference."""
        print(f"DEBUG: Processing {len(answers)} answers with complete Mamdani fuzzy logic")
        
        # Step 1: Calculate symptom intensities from answers
        symptom_intensities = self._calculate_symptom_intensities(answers)
        print(f"DEBUG: Calculated {len(symptom_intensities)} symptom intensities")
        
        scores = {}
        aggregated_severities = {}
        
        # Step 2: Process each disorder
        for disorder in self.disorder_criteria:
            print(f"\nDEBUG: Processing {disorder} with Mamdani inference")
            
            # Special handling for ASPD age requirement
            if disorder == "Antisocial" and "aspd_q9" in answers:
                if answers["aspd_q9"].lower() != "yes":
                    print(f"DEBUG: ASPD age requirement not met - setting probability to 0")
                    scores[disorder] = 0.0
                    aggregated_severities[disorder] = 0.0
                    continue
            
            # Step 2a: Aggregate symptoms for this disorder
            avg_intensity, consistency, criteria_ratio = self._aggregate_symptoms_by_disorder(
                symptom_intensities, disorder
            )
            print(f"DEBUG: {disorder} - Avg Intensity: {avg_intensity:.1f}, "
                  f"Consistency: {consistency:.1f}, Criteria Ratio: {criteria_ratio:.3f}")
            
            # Step 2b: Apply fuzzy symptom aggregation
            aggregated_severity = self._apply_symptom_aggregation(
                disorder, avg_intensity, consistency, criteria_ratio
            )
            aggregated_severities[disorder] = aggregated_severity
            print(f"DEBUG: {disorder} - Aggregated Severity: {aggregated_severity:.1f}")
            
            # Step 2c: Calculate clinical threshold satisfaction
            threshold_ratio = self._calculate_clinical_threshold_ratio(answers, disorder)
            print(f"DEBUG: {disorder} - Clinical Threshold Satisfaction: {threshold_ratio:.3f}")
            
            # Step 2d: Calculate final disorder probability
            probability = self._calculate_final_disorder_probability(
                disorder, aggregated_severity, threshold_ratio
            )
            scores[disorder] = round(probability, 1)
            print(f"DEBUG: {disorder} - Final Probability: {probability:.1f}%")
        
        # Step 3: Apply comorbidity adjustments
        adjusted_scores = self._apply_comorbidity_adjustments(scores, aggregated_severities)
        
        print(f"\nDEBUG: Final Mamdani scores: {adjusted_scores}")
        return adjusted_scores
    
    def _apply_comorbidity_adjustments(self, base_scores, aggregated_severities):
        """Apply fuzzy comorbidity adjustments to all disorder pairs."""
        adjusted_scores = base_scores.copy()
        
        # Define known comorbidity patterns with interaction strengths
        comorbidity_patterns = {
            ("Borderline", "Narcissistic"): 0.8,
            ("Borderline", "Histrionic"): 0.7,
            ("Narcissistic", "Histrionic"): 0.6,
            ("Borderline", "Antisocial"): 0.5,
            ("Narcissistic", "Antisocial"): 0.7,
            ("Histrionic", "Antisocial"): 0.4
        }
        
        # Apply adjustments for each known comorbidity pattern
        for (disorder1, disorder2), interaction_strength in comorbidity_patterns.items():
            if disorder1 in base_scores and disorder2 in base_scores:
                prob1 = base_scores[disorder1]
                prob2 = base_scores[disorder2]
                
                # Only apply adjustment if both disorders have significant presence
                if prob1 >= 20 and prob2 >= 20:
                    adjustment_factor = self._apply_comorbidity_adjustment(
                        disorder1, disorder2, prob1, prob2, interaction_strength
                    )
                    
                    # Apply adjustment to both disorders
                    adjusted_scores[disorder1] = min(85.0, prob1 * adjustment_factor)
                    adjusted_scores[disorder2] = min(85.0, prob2 * adjustment_factor)
                    
                    print(f"DEBUG: Comorbidity adjustment for {disorder1}-{disorder2}: "
                          f"factor={adjustment_factor:.3f}")
        
        return adjusted_scores
    
    def _calculate_risk_levels(self, answers, scores):
        """Calculate risk levels using fuzzy logic."""
        # Initialize risk components
        self_harm_risk = 0.0
        aggression_risk = 0.0
        
        # Calculate self-harm risk from specific questions
        if "bpd_q5" in answers:  # Self-harm/suicidal behavior
            intensity = self._calculate_symptom_intensities({"bpd_q5": answers["bpd_q5"]})
            self_harm_risk = intensity.get("bpd_q5", 0.0)
        
        # Calculate aggression risk from specific questions
        aggression_questions = ["aspd_q4", "aspd_q5", "bpd_q8"]
        aggression_intensities = []
        for q in aggression_questions:
            if q in answers:
                intensity = self._calculate_symptom_intensities({q: answers[q]})
                aggression_intensities.append(intensity.get(q, 0.0))
        
        if aggression_intensities:
            aggression_risk = max(aggression_intensities)  # Use highest risk indicator
        
        # Calculate overall severity from disorder scores
        overall_severity = max(scores.values()) if scores else 0.0
        
        # Apply fuzzy risk assessment
        try:
            simulator = self.risk_assessment_system['simulator']
            
            # Set inputs
            simulator.input['self_harm_risk'] = self_harm_risk
            simulator.input['aggression_risk'] = aggression_risk
            simulator.input['overall_severity'] = overall_severity
            
            # Compute
            simulator.compute()
            
            # Get risk level
            risk_score = simulator.output['risk_level']
            
            # Convert to qualitative level
            if risk_score >= 75:
                risk_level = "Critical"
            elif risk_score >= 50:
                risk_level = "High"
            elif risk_score >= 30:
                risk_level = "Moderate"
            else:
                risk_level = "Low"
            
            return {
                "level": risk_level,
                "score": risk_score,
                "components": {
                    "self_harm": self_harm_risk,
                    "aggression": aggression_risk,
                    "overall_severity": overall_severity
                }
            }
            
        except Exception as e:
            print(f"Error in risk assessment: {e}")
            return {
                "level": "Unknown",
                "score": 0.0,
                "components": {}
            }
    
    def generate_enhanced_recommendations(self, scores):
        """Generate recommendations using fuzzy logic insights."""
        recommendations = []
        
        # Sort disorders by score
        sorted_disorders = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        for disorder, score in sorted_disorders:
            if score >= 10:
                # Determine confidence level using fuzzy logic
                confidence_level = self._determine_confidence_level(score)
                
                # Get disorder info from knowledge base
                disorder_info = self.kb.get_disorder_info(disorder) if hasattr(self.kb, 'get_disorder_info') else None
                
                rec = {
                    "disorder": disorder,
                    "score": round(score, 1),
                    "confidence_level": confidence_level,
                    "description": self._generate_description(disorder, score, confidence_level),
                    "criteria": disorder_info["criteria"] if disorder_info else [],
                    "action_steps": self._get_action_steps(disorder, score, confidence_level),
                    "clinical_significance": self._assess_clinical_significance(score)
                }
                
                recommendations.append(rec)
        
        return recommendations
    
    def _determine_confidence_level(self, score):
        """Determine confidence level using fuzzy membership."""
        # Define fuzzy sets for confidence levels
        if score >= 70:
            if score >= 85:
                return "Very High"
            return "High"
        elif score >= 50:
            return "Moderate-High"
        elif score >= 30:
            return "Moderate"
        elif score >= 15:
            return "Low-Moderate"
        else:
            return "Low"
    
    def _generate_description(self, disorder, score, confidence_level):
        """Generate description based on fuzzy confidence level."""
        templates = {
            "Very High": f"Very strong indication of {disorder} Personality Disorder traits",
            "High": f"Strong indication of {disorder} Personality Disorder traits",
            "Moderate-High": f"Notable presence of {disorder} Personality Disorder characteristics",
            "Moderate": f"Moderate likelihood of {disorder} Personality Disorder features",
            "Low-Moderate": f"Some evidence of {disorder} Personality Disorder traits",
            "Low": f"Minimal indication of {disorder} Personality Disorder patterns"
        }
        return templates.get(confidence_level, f"{disorder} Personality Disorder traits detected")
    
    def _get_action_steps(self, disorder, score, confidence_level):
        """Get action steps based on disorder and confidence level."""
        steps = []
        
        # Universal recommendations based on confidence
        if confidence_level in ["Very High", "High"]:
            steps.extend([
                "Seek immediate professional evaluation from a licensed mental health professional",
                "Consider specialized assessment for personality disorders",
                "Explore evidence-based therapeutic interventions"
            ])
        elif confidence_level in ["Moderate-High", "Moderate"]:
            steps.extend([
                "Schedule consultation with a mental health professional",
                "Consider psychological assessment for personality traits",
                "Monitor symptoms and their impact on daily functioning"
            ])
        else:
            steps.extend([
                "Consider self-monitoring of identified traits",
                "Explore psychoeducational resources about personality patterns",
                "Maintain awareness of symptom development"
            ])
        
        # Disorder-specific recommendations
        disorder_specific = {
            "Borderline": [
                "Research Dialectical Behavior Therapy (DBT) options",
                "Practice mindfulness and emotional regulation techniques"
            ],
            "Antisocial": [
                "Consider Cognitive Behavioral Therapy (CBT)",
                "Work on developing empathy and social awareness"
            ],
            "Histrionic": [
                "Explore Psychodynamic or Schema Therapy approaches",
                "Work on developing authentic emotional expression"
            ],
            "Narcissistic": [
                "Consider specialized therapy for narcissistic traits",
                "Work on developing genuine empathy and self-awareness"
            ]
        }
        
        if disorder in disorder_specific:
            steps.extend(disorder_specific[disorder])
        
        return steps
    
    def _assess_clinical_significance(self, score):
        """Assess clinical significance using fuzzy logic."""
        # Use fuzzy membership to determine significance
        if score >= 80:
            return "Very High"
        elif score >= 65:
            return "High"
        elif score >= 45:
            return "Moderate"
        elif score >= 25:
            return "Low-Moderate"
        else:
            return "Low"
    
    def identify_comorbidity_patterns(self, scores):
        """Identify comorbidity patterns using fuzzy logic."""
        patterns = []
        
        # Get disorders with significant scores
        significant_disorders = [(d, s) for d, s in scores.items() if s >= 20.0]
        
        if len(significant_disorders) < 2:
            return patterns
        
        # Known comorbidity relationships with fuzzy interaction strengths
        known_patterns = {
            ("Borderline", "Narcissistic"): {
                "interaction_strength": 0.8,
                "shared_features": ["identity_disturbance", "anger_issues", "unstable_relationships"]
            },
            ("Borderline", "Histrionic"): {
                "interaction_strength": 0.7,
                "shared_features": ["attention_seeking", "mood_instability", "theatrical"]
            },
            ("Narcissistic", "Histrionic"): {
                "interaction_strength": 0.6,
                "shared_features": ["attention_seeking", "grandiosity", "shallow_emotions"]
            },
            ("Borderline", "Antisocial"): {
                "interaction_strength": 0.5,
                "shared_features": ["impulsivity", "anger_issues", "unstable_relationships"]
            },
            ("Narcissistic", "Antisocial"): {
                "interaction_strength": 0.7,
                "shared_features": ["lacks_empathy", "interpersonally_exploitative", "grandiosity"]
            },
            ("Histrionic", "Antisocial"): {
                "interaction_strength": 0.4,
                "shared_features": ["attention_seeking", "shallow_emotions"]
            }
        }
        
        # Check all combinations
        for (disorder1, score1), (disorder2, score2) in combinations(significant_disorders, 2):
            pattern_key = tuple(sorted([disorder1, disorder2]))
            
            if pattern_key in known_patterns:
                pattern_info = known_patterns[pattern_key]
                
                # Calculate clinical significance using fuzzy logic
                avg_score = (score1 + score2) / 2
                min_score = min(score1, score2)
                
                # Fuzzy calculation of significance
                base_significance = avg_score * 0.7 + min_score * 0.3
                interaction_boost = pattern_info["interaction_strength"] * 15
                
                clinical_significance = min(100.0, base_significance + interaction_boost)
                
                if clinical_significance >= 25.0:
                    patterns.append({
                        "disorders": [disorder1, disorder2],
                        "scores": [round(score1, 1), round(score2, 1)],
                        "clinical_significance": round(clinical_significance, 1),
                        "interaction_strength": pattern_info["interaction_strength"],
                        "shared_features": pattern_info["shared_features"]
                    })
        
        # Sort by clinical significance
        patterns.sort(key=lambda x: x["clinical_significance"], reverse=True)
        
        return patterns
    
    def generate_detailed_assessment_report(self, scores, answers):
        """Generate comprehensive assessment report using fuzzy logic."""
        recommendations = self.generate_enhanced_recommendations(scores)
        comorbid_patterns = self.identify_comorbidity_patterns(scores)
        risk_assessment = self._calculate_risk_levels(answers, scores)
        
        # Overall assessment
        overall_assessment = self._calculate_overall_assessment(scores, comorbid_patterns)
        
        report = {
            "assessment_summary": {
                "overall_significance": overall_assessment["significance"],
                "confidence_level": overall_assessment["confidence"],
                "primary_concerns": overall_assessment["primary_concerns"],
                "risk_level": risk_assessment["level"]
            },
            "disorder_analysis": recommendations,
            "comorbidity_patterns": comorbid_patterns,
            "clinical_recommendations": self._generate_clinical_recommendations(
                recommendations, comorbid_patterns, risk_assessment
            ),
            "monitoring_suggestions": self._generate_monitoring_suggestions(scores),
            "resource_recommendations": self._generate_resource_recommendations(recommendations)
        }
        
        return report
    
    def _calculate_overall_assessment(self, scores, comorbid_patterns):
        """Calculate overall assessment using fuzzy aggregation."""
        if not scores:
            return {"significance": "None", "confidence": "N/A", "primary_concerns": []}
        
        # Get top scores
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_score = sorted_scores[0][1] if sorted_scores else 0
        
        # Fuzzy determination of significance
        comorbidity_factor = len(comorbid_patterns)
        
        # Apply fuzzy rules for overall significance
        if comorbidity_factor >= 2 or top_score >= 80:
            significance_level = "Very High"
            confidence = "High"
        elif comorbidity_factor == 1 and top_score >= 50:
            significance_level = "High"
            confidence = "Moderate-High"
        elif top_score >= 60:
            significance_level = "High"
            confidence = "Moderate-High"
        elif top_score >= 40:
            significance_level = "Moderate"
            confidence = "Moderate"
        elif top_score >= 25:
            significance_level = "Low-Moderate"
            confidence = "Low-Moderate"
        else:
            significance_level = "Low"
            confidence = "Low"
        
        # Identify primary concerns
        primary_concerns = [disorder for disorder, score in sorted_scores[:3] if score >= 25]
        
        return {
            "significance": significance_level,
            "confidence": confidence,
            "primary_concerns": primary_concerns
        }
    
    def _generate_clinical_recommendations(self, disorder_recommendations, comorbid_patterns, risk_assessment):
        """Generate clinical recommendations using fuzzy logic insights."""
        recommendations = []
        
        # Priority based on risk level
        if risk_assessment["level"] in ["Critical", "High"]:
            recommendations.extend([
                "URGENT: Seek immediate professional mental health evaluation",
                "Consider crisis intervention services if experiencing acute distress",
                "Ensure safety planning is in place"
            ])
        
        # Primary treatment recommendations
        if disorder_recommendations:
            primary_disorder = disorder_recommendations[0]["disorder"]
            recommendations.append(f"Prioritize assessment and treatment for {primary_disorder} traits")
        
        # Comorbidity-specific recommendations
        if len(comorbid_patterns) >= 2:
            recommendations.extend([
                "Seek specialist evaluation for complex personality presentations",
                "Consider integrated treatment approach addressing multiple trait patterns"
            ])
        elif len(comorbid_patterns) == 1:
            recommendations.append("Consider comprehensive assessment for co-occurring personality traits")
        
        # Evidence-based treatments
        for rec in disorder_recommendations[:2]:
            if rec["confidence_level"] in ["Very High", "High", "Moderate-High"]:
                disorder = rec["disorder"]
                if disorder == "Borderline":
                    recommendations.append("Dialectical Behavior Therapy (DBT) - gold standard for borderline traits")
                elif disorder == "Antisocial":
                    recommendations.append("Cognitive Behavioral Therapy (CBT) with focus on behavioral change")
                elif disorder == "Histrionic":
                    recommendations.append("Psychodynamic therapy for emotional regulation")
                elif disorder == "Narcissistic":
                    recommendations.append("Schema-Focused Therapy for narcissistic patterns")
        
        return recommendations
    
    def _generate_monitoring_suggestions(self, scores):
        """Generate monitoring suggestions based on fuzzy assessment."""
        suggestions = []
        
        # Add monitoring based on significant scores
        for disorder, score in scores.items():
            if score >= 30:
                if disorder == "Borderline":
                    suggestions.extend([
                        "Daily mood tracking with intensity ratings",
                        "Monitor interpersonal conflict patterns"
                    ])
                elif disorder == "Antisocial":
                    suggestions.extend([
                        "Monitor aggressive impulses and triggers",
                        "Track compliance with rules and responsibilities"
                    ])
                elif disorder == "Histrionic":
                    suggestions.extend([
                        "Monitor attention-seeking behaviors",
                        "Track emotional intensity and duration"
                    ])
                elif disorder == "Narcissistic":
                    suggestions.extend([
                        "Monitor reactions to criticism or feedback",
                        "Track empathetic responses to others"
                    ])
        
        # General monitoring
        suggestions.extend([
            "Weekly self-assessment using validated screening tools",
            "Regular check-ins with mental health professionals"
        ])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_suggestions = []
        for item in suggestions:
            if item not in seen:
                seen.add(item)
                unique_suggestions.append(item)
        
        return unique_suggestions
    
    def _generate_resource_recommendations(self, disorder_recommendations):
        """Generate resource recommendations."""
        resources = {
            "educational": ["Personality disorder psychoeducation materials"],
            "support_groups": ["NAMI (National Alliance on Mental Illness) support groups"],
            "crisis_resources": [
                "National Suicide Prevention Lifeline: 988",
                "Crisis Text Line: Text HOME to 741741"
            ],
            "self_help": ["Mindfulness and emotional regulation apps"]
        }
        
        # Add disorder-specific resources
        for rec in disorder_recommendations:
            disorder = rec["disorder"]
            if disorder == "Borderline":
                resources["educational"].append("DBT Skills Training Manual")
                resources["support_groups"].append("Borderline Personality Disorder support groups")
            elif disorder == "Antisocial":
                resources["educational"].append("CBT workbooks for impulse control")
            elif disorder == "Histrionic":
                resources["educational"].append("Emotional regulation psychoeducation")
            elif disorder == "Narcissistic":
                resources["educational"].append("Empathy development resources")
        
        return resources