class ClusterBKnowledgeBase:
    """
    Knowledge base for Cluster B personality disorders using fuzzy logic principles.
    Contains symptoms and criteria for Borderline, Antisocial, Histrionic, and Narcissistic
    personality disorders according to DSM diagnostic criteria.
    """
    
    def __init__(self):
        # Define the four Cluster B personality disorders
        self.disorders = {
            "Borderline": {"min_criteria": 5, "total_criteria": 9},
            "Antisocial": {"min_criteria": 3, "total_criteria": 7, "age_requirement": 18, "conduct_disorder_before_15": True},
            "Histrionic": {"min_criteria": 5, "total_criteria": 8},
            "Narcissistic": {"min_criteria": 5, "total_criteria": 9}
        }
        
        # Initialize the knowledge base with criteria and symptoms for each disorder
        self.criteria = self._initialize_criteria()
        
        # Questions mapped to criteria for the inference engine
        self.questions = self._initialize_questions()
        
        # Mapping between questions and the criteria they assess
        self.question_to_criteria_map = self._initialize_question_mapping()
    
    def _initialize_criteria(self):
        """Define criteria for each disorder with detailed symptom descriptions"""
        
        criteria = {
            "Borderline": {
                "abandonment_fear": {
                    "name": "Frantic efforts to avoid abandonment",
                    "description": "Intense fear of being abandoned, with extreme reactions to real or perceived abandonment",
                    "indicators": [
                        "Desperate reactions to perceived separation",
                        "Intense distress with changes in plans",
                        "Panic or anger when someone important is late",
                        "Believing abandonment implies being 'bad'",
                        "Intolerance of being alone",
                        "Impulsive actions to avoid abandonment"
                    ]
                },
                "unstable_relationships": {
                    "name": "Unstable and intense relationships",
                    "description": "Pattern of unstable relationships alternating between idealization and devaluation",
                    "indicators": [
                        "Rapid idealization of new relationships",
                        "Quick shifts to devaluing others",
                        "Dramatic view shifts of others",
                        "Expecting others to be available on demand",
                        "Intense expectations in relationships"
                    ]
                },
                "identity_disturbance": {
                    "name": "Identity disturbance",
                    "description": "Markedly and persistently unstable self-image or sense of self",
                    "indicators": [
                        "Sudden shifts in self-image",
                        "Changing goals, values, and aspirations",
                        "Shifting between needy and vengeful roles",
                        "Feelings of not existing",
                        "Poor performance in unstructured settings"
                    ]
                },
                "self_damaging_impulsivity": {
                    "name": "Self-damaging impulsivity",
                    "description": "Impulsivity in at least two potentially self-damaging areas",
                    "indicators": [
                        "Gambling",
                        "Spending money irresponsibly",
                        "Binge eating",
                        "Substance abuse",
                        "Unsafe sex",
                        "Reckless driving"
                    ]
                },
                "suicidal_behavior": {
                    "name": "Suicidal behavior or self-harm",
                    "description": "Recurrent suicidal behavior, gestures, threats, or self-mutilating behavior",
                    "indicators": [
                        "Suicide attempts",
                        "Self-mutilation (cutting, burning)",
                        "Suicide threats",
                        "Self-destructive acts during stress",
                        "Relief from self-harm"
                    ]
                },
                "mood_instability": {
                    "name": "Affective instability",
                    "description": "Affective instability due to marked reactivity of mood",
                    "indicators": [
                        "Intense episodic dysphoria",
                        "Irritability",
                        "Anxiety episodes",
                        "Extreme reactions to interpersonal stress",
                        "Rapid mood fluctuations"
                    ]
                },
                "emptiness": {
                    "name": "Chronic emptiness",
                    "description": "Chronic feelings of emptiness",
                    "indicators": [
                        "Easily bored",
                        "Constant need for activity",
                        "Feeling empty inside",
                        "Lacking sense of fulfillment"
                    ]
                },
                "anger_issues": {
                    "name": "Inappropriate anger",
                    "description": "Inappropriate, intense anger or difficulty controlling anger",
                    "indicators": [
                        "Frequent displays of temper",
                        "Constant anger",
                        "Physical fights",
                        "Extreme sarcasm",
                        "Verbal outbursts",
                        "Anger followed by shame and guilt"
                    ]
                },
                "paranoid_dissociation": {
                    "name": "Paranoid ideation or dissociation",
                    "description": "Transient, stress-related paranoid ideation or severe dissociative symptoms",
                    "indicators": [
                        "Paranoid thoughts during stress",
                        "Dissociative episodes",
                        "Symptoms occur with abandonment fears",
                        "Transient symptoms (minutes to hours)"
                    ]
                }
            },
            
            "Antisocial": {
                "law_violation": {
                    "name": "Failure to conform to social norms",
                    "description": "Repeated acts that are grounds for arrest",
                    "indicators": [
                        "Destroying property",
                        "Harassing others",
                        "Stealing",
                        "Illegal occupations",
                        "Disregard for others' rights"
                    ]
                },
                "deceitfulness": {
                    "name": "Deceitfulness",
                    "description": "Repeated lying, use of aliases, or conning others",
                    "indicators": [
                        "Pathological lying",
                        "Using aliases",
                        "Conning others for profit",
                        "Conning others for pleasure"
                    ]
                },
                "impulsivity": {
                    "name": "Impulsivity or failure to plan",
                    "description": "Acting on the spur of the moment without forethought",
                    "indicators": [
                        "Spontaneous decisions without planning",
                        "Disregard for consequences",
                        "Sudden job changes",
                        "Frequent relocation",
                        "Unstable relationships"
                    ]
                },
                "aggressiveness": {
                    "name": "Irritability and aggressiveness",
                    "description": "Repeated physical fights or assaults",
                    "indicators": [
                        "Physical fights",
                        "Assaults",
                        "Domestic violence",
                        "Child abuse"
                    ]
                },
                "reckless_disregard": {
                    "name": "Reckless disregard for safety",
                    "description": "Disregard for safety of self or others",
                    "indicators": [
                        "Reckless driving",
                        "DUIs",
                        "Multiple accidents",
                        "High-risk sexual behavior",
                        "Substance abuse",
                        "Child endangerment"
                    ]
                },
                "irresponsibility": {
                    "name": "Consistent irresponsibility",
                    "description": "Failure to sustain work or honor financial obligations",
                    "indicators": [
                        "Employment instability despite opportunities",
                        "Job abandonment",
                        "Unexplained work absences",
                        "Defaulting on debts",
                        "Failing to provide child support"
                    ]
                },
                "lack_of_remorse": {
                    "name": "Lack of remorse",
                    "description": "Indifference to or rationalization of having hurt others",
                    "indicators": [
                        "Indifference to hurting others",
                        "Rationalizing harmful actions",
                        "Blaming victims",
                        "Minimizing harmful consequences",
                        "Failure to make amends"
                    ]
                },
                "conduct_disorder": {
                    "name": "Conduct disorder before age 15",
                    "description": "Evidence of conduct disorder with onset before age 15",
                    "indicators": [
                        "Aggression to people and animals in childhood",
                        "Destruction of property in childhood",
                        "Deceitfulness or theft in childhood",
                        "Serious violation of rules in childhood"
                    ]
                }
            },
            
            "Histrionic": {
                "attention_seeking": {
                    "name": "Uncomfortable when not center of attention",
                    "description": "Discomfort in situations where not the center of attention",
                    "indicators": [
                        "Drawing attention to self",
                        "Creating dramatic scenes",
                        "Enthusiastic or flirtatious to get attention",
                        "Dramatic descriptions of symptoms",
                        "Attention-seeking behavior with clinicians"
                    ]
                },
                "sexual_seductiveness": {
                    "name": "Inappropriate seductiveness",
                    "description": "Inappropriate sexually seductive or provocative behavior",
                    "indicators": [
                        "Flirtation in inappropriate contexts",
                        "Provocative behavior in professional settings",
                        "Seductive behavior beyond social context norms"
                    ]
                },
                "shallow_emotions": {
                    "name": "Rapidly shifting emotions",
                    "description": "Rapidly shifting and shallow expression of emotions",
                    "indicators": [
                        "Quick emotional changes",
                        "Emotions appear shallow",
                        "Emotions seem turned on/off easily",
                        "Apparent fake feelings"
                    ]
                },
                "appearance_focus": {
                    "name": "Uses appearance for attention",
                    "description": "Consistently uses physical appearance to draw attention",
                    "indicators": [
                        "Excessive concern with appearance",
                        "Significant time/money on grooming",
                        "Fishing for compliments",
                        "Easily upset by criticism of appearance"
                    ]
                },
                "impressionistic_speech": {
                    "name": "Impressionistic speech",
                    "description": "Speech that is excessively impressionistic and lacking in detail",
                    "indicators": [
                        "Vague opinions",
                        "Dramatic flair without substance",
                        "Lacking supporting facts",
                        "Imprecise communication"
                    ]
                },
                "theatrical": {
                    "name": "Self-dramatization",
                    "description": "Shows self-dramatization, theatricality, and exaggerated expression of emotion",
                    "indicators": [
                        "Excessive public display of emotions",
                        "Emotional overreactions",
                        "Temper tantrums",
                        "Dramatic expressions"
                    ]
                },
                "suggestibility": {
                    "name": "Suggestibility",
                    "description": "Easily influenced by others or circumstances",
                    "indicators": [
                        "Opinions easily influenced",
                        "Susceptible to fads",
                        "Overly trusting of authority figures",
                        "Quick adoption of convictions"
                    ]
                },
                "misinterprets_relationships": {
                    "name": "Misinterprets relationship intimacy",
                    "description": "Considers relationships to be more intimate than they actually are",
                    "indicators": [
                        "Overestimates closeness of relationships",
                        "Acting out roles in relationships",
                        "Confusing acquaintances for close friends",
                        "Misinterpreting professional relationships"
                    ]
                }
            },
            
            "Narcissistic": {
                "grandiosity": {
                    "name": "Grandiose self-importance",
                    "description": "Grandiose sense of self-importance",
                    "indicators": [
                        "Exaggerates achievements",
                        "Overestimates abilities",
                        "Appears boastful",
                        "Expects recognition without achievement",
                        "Devalues others' contributions"
                    ]
                },
                "fantasy_preoccupation": {
                    "name": "Fantasy preoccupation",
                    "description": "Preoccupied with fantasies of unlimited success, power, brilliance, beauty, or ideal love",
                    "indicators": [
                        "Rumination about deserved admiration",
                        "Comparing self to famous people",
                        "Fantasies of greatness",
                        "Preoccupation with perfect scenarios"
                    ]
                },
                "believes_special": {
                    "name": "Believes they are special",
                    "description": "Believes they are special and unique and can only associate with other special people",
                    "indicators": [
                        "Only understood by special people",
                        "Should only associate with high-status people",
                        "Assigns unique qualities to associates",
                        "Believes needs are beyond ordinary people",
                        "Insists on having only the 'top' professionals"
                    ]
                },
                "requires_admiration": {
                    "name": "Requires excessive admiration",
                    "description": "Requires excessive admiration",
                    "indicators": [
                        "Fragile self-esteem",
                        "Preoccupied with how they're perceived",
                        "Needs constant attention",
                        "Expects great recognition",
                        "Constantly seeks compliments"
                    ]
                },
                "entitlement": {
                    "name": "Sense of entitlement",
                    "description": "Unreasonable expectations of favorable treatment",
                    "indicators": [
                        "Expects to be catered to",
                        "Angry when not given special treatment",
                        "Expects automatic compliance",
                        "Unreasonable expectations"
                    ]
                },
                "interpersonally_exploitative": {
                    "name": "Interpersonally exploitative",
                    "description": "Takes advantage of others to achieve own ends",
                    "indicators": [
                        "Expects to be given what they want",
                        "Forms relationships for personal gain",
                        "Uses others to enhance self-esteem",
                        "Takes special privileges"
                    ]
                },
                "lacks_empathy": {
                    "name": "Lacks empathy",
                    "description": "Unwilling to recognize or identify with feelings of others",
                    "indicators": [
                        "Assumes others focused on their welfare",
                        "Discusses own concerns excessively",
                        "Impatient with others' problems",
                        "Oblivious to hurt caused",
                        "Views others' needs as weakness"
                    ]
                },
                "envious": {
                    "name": "Envious of others",
                    "description": "Envious of others or believes others are envious of them",
                    "indicators": [
                        "Begrudges others' success",
                        "Feels more deserving of recognition",
                        "Devalues others' achievements",
                        "Believes others envy them"
                    ]
                },
                "arrogant": {
                    "name": "Arrogant behavior",
                    "description": "Shows arrogant, haughty behaviors or attitudes",
                    "indicators": [
                        "Displays superiority",
                        "Condescending attitude",
                        "Looks down on others",
                        "Reacts to criticism with rage",
                        "Appears disdainful"
                    ]
                }
            }
        }
        
        return criteria
    
    def _initialize_questions(self):
        """Initialize assessment questions for each disorder criteria based on the provided document"""
        
        questions = {
            # Borderline PD Questions from document
            "bpd_q1": "Do you go out of your way to avoid feeling abandoned, even when the threat may not be clear (e.g., feeling anxious when plans change or someone is late)?",
            "bpd_q2": "Do your close relationships tend to swing between feeling very close and then you feeling let down or upset?",
            "bpd_q3": "Do you experience sudden shifts in your goals, values, or sense of identity?",
            "bpd_q4": "Do you engage in impulsive behaviors that could harm you (e.g., unsafe sex, binge eating, reckless driving, substance abuse)?",
            "bpd_q5": "Do you ever have recurring thoughts of not wanting to live, or have you ever hurt yourself when feeling overwhelmed?",
            "bpd_q6": "Do your emotions tend to change quickly and strongly in response to relationship stress or conflict? (e.g., going from calm to despair within hours)?",
            "bpd_q7": "Do you often feel chronically empty or bored, like something is missing inside you?",
            "bpd_q8": "Do you sometimes find it difficult to manage your anger or frustration (e.g., snapping at people, feeling intense rage)?",
            "bpd_q9": "Under stress, do you ever feel disconnected from reality or suspicious of others, especially in times of conflict or fear?",
            
            # Antisocial PD Questions from document
            "aspd_q1": "Since your teenage years, have you had challenges following rules or laws (e.g., getting into trouble at school, work, or with the law)?",
            "aspd_q2": "Do you sometimes stretch the truth or present yourself differently to get through a situation or meet your needs?",
            "aspd_q3": "Do you act on impulse without thinking through the long-term consequences (e.g., quitting jobs suddenly, reckless decisions)?",
            "aspd_q4": "Have you struggled with irritability or frustration that sometimes leads to arguments or physical confrontations?",
            "aspd_q5": "Do you often act without concern for your own or others' safety (e.g., drunk driving, risky sex, neglecting children)?",
            "aspd_q6": "Before age 15, did you often get into trouble for fighting, lying, or breaking rules at home or school?",
            "aspd_q7": "Do you feel little or no remorse when you harm others or break rules (e.g., rationalizing or blaming the victim)?",
            "aspd_q8": "Before age 15, did you often get into trouble for fighting, lying, or breaking rules at home or school?",
            "aspd_q9": "Are you currently over the age of 18?",
            
            # Histrionic PD Questions from document
            "hpd_q1": "Do you feel uncomfortable when you're not the center of attention and try to regain it (e.g., being dramatic, exaggerating stories)?",
            "hpd_q2": "In social interactions, do you sometimes find yourself being more flirtatious or expressive than others might expect?",
            "hpd_q3": "Do your emotions tend to shift quickly and feel very intense, even in minor situations?",
            "hpd_q4": "Do you put a lot of effort into your appearance or behavior to make a strong impression?",
            "hpd_q5": "Is your style of talking sometimes vague or filled with expressive language without much detail?",
            "hpd_q6": "Do you sometimes react very strongly to everyday events (e.g., crying easily, celebrating enthusiastically)?",
            "hpd_q7": "Do you often adopt opinions or behaviors that match those around you?",
            "hpd_q8": "Do you believe relationships are more intimate or close than they actually are (e.g., thinking a casual acquaintance is a close friend)?",
            
            # Narcissistic PD Questions from document
            "npd_q1": "Do you feel most comfortable with people you see as unique, successful, or influential?",
            "npd_q2": "Are you preoccupied with fantasies of unlimited success, power, beauty, brilliance, or perfect love?",
            "npd_q3": "Do you feel most comfortable with people you see as unique, successful, or influential?",
            "npd_q4": "Do you require excessive admiration or get upset if you don't receive it (e.g., fishing for compliments, reacting poorly to criticism)?",
            "npd_q5": "Do you sometimes expect others to meet your needs without needing to explain why?",
            "npd_q6": "Do you exploit others to achieve your own goals (e.g., using people to get ahead without considering their needs)?",
            "npd_q7": "Do you struggle to recognize or care about other people's feelings or needs (e.g., focus only on your own problems)?",
            "npd_q8": "Do you occasionally feel jealous of others or think they might be jealous of you?",
            "npd_q9": "Have others ever described you as confident to the point of seeming boastful?"
        }
        
        return questions
    
    def _initialize_question_mapping(self):
        """Map questions to their relevant criteria for each disorder"""
        
        mapping = {
            # Borderline PD mappings
            "bpd_q1": {"disorder": "Borderline", "criterion": "abandonment_fear"},
            "bpd_q2": {"disorder": "Borderline", "criterion": "unstable_relationships"},
            "bpd_q3": {"disorder": "Borderline", "criterion": "identity_disturbance"},
            "bpd_q4": {"disorder": "Borderline", "criterion": "self_damaging_impulsivity"},
            "bpd_q5": {"disorder": "Borderline", "criterion": "suicidal_behavior"},
            "bpd_q6": {"disorder": "Borderline", "criterion": "mood_instability"},
            "bpd_q7": {"disorder": "Borderline", "criterion": "emptiness"},
            "bpd_q8": {"disorder": "Borderline", "criterion": "anger_issues"},
            "bpd_q9": {"disorder": "Borderline", "criterion": "paranoid_dissociation"},
            
            # Antisocial PD mappings
            "aspd_q1": {"disorder": "Antisocial", "criterion": "law_violation"},
            "aspd_q2": {"disorder": "Antisocial", "criterion": "deceitfulness"},
            "aspd_q3": {"disorder": "Antisocial", "criterion": "impulsivity"},
            "aspd_q4": {"disorder": "Antisocial", "criterion": "aggressiveness"},
            "aspd_q5": {"disorder": "Antisocial", "criterion": "reckless_disregard"},
            "aspd_q6": {"disorder": "Antisocial", "criterion": "irresponsibility"},
            "aspd_q7": {"disorder": "Antisocial", "criterion": "lack_of_remorse"},
            "aspd_q8": {"disorder": "Antisocial", "criterion": "conduct_disorder"},
            "aspd_q9": {"disorder": "Antisocial", "criterion": "age_requirement"},
            
            # Histrionic PD mappings
            "hpd_q1": {"disorder": "Histrionic", "criterion": "attention_seeking"},
            "hpd_q2": {"disorder": "Histrionic", "criterion": "sexual_seductiveness"},
            "hpd_q3": {"disorder": "Histrionic", "criterion": "shallow_emotions"},
            "hpd_q4": {"disorder": "Histrionic", "criterion": "appearance_focus"},
            "hpd_q5": {"disorder": "Histrionic", "criterion": "impressionistic_speech"},
            "hpd_q6": {"disorder": "Histrionic", "criterion": "theatrical"},
            "hpd_q7": {"disorder": "Histrionic", "criterion": "suggestibility"},
            "hpd_q8": {"disorder": "Histrionic", "criterion": "misinterprets_relationships"},
            
            # Narcissistic PD mappings
            "npd_q1": {"disorder": "Narcissistic", "criterion": "grandiosity"},
            "npd_q2": {"disorder": "Narcissistic", "criterion": "fantasy_preoccupation"},
            "npd_q3": {"disorder": "Narcissistic", "criterion": "believes_special"},
            "npd_q4": {"disorder": "Narcissistic", "criterion": "requires_admiration"},
            "npd_q5": {"disorder": "Narcissistic", "criterion": "entitlement"},
            "npd_q6": {"disorder": "Narcissistic", "criterion": "interpersonally_exploitative"},
            "npd_q7": {"disorder": "Narcissistic", "criterion": "lacks_empathy"},
            "npd_q8": {"disorder": "Narcissistic", "criterion": "envious"},
            "npd_q9": {"disorder": "Narcissistic", "criterion": "arrogant"}
        }
        
        return mapping
    
    def get_question(self, question_id):
        """Get question information including its impacts on disorders."""
        if question_id not in self.questions:
            return None
            
        question_text = self.questions[question_id]
        impacts = {}
        
        # Get the mapping for this question
        if question_id in self.question_to_criteria_map:
            mapping = self.question_to_criteria_map[question_id]
            disorder = mapping["disorder"]
            criterion = mapping["criterion"]
            
            # Set impact based on the criterion's importance
            if criterion in self.criteria[disorder]:
                impacts[disorder] = 1.0  # Base impact
        
        return {
            "id": question_id,
            "text": question_text,
            "impacts": impacts
        }
    
    def get_criterion_info(self, disorder, criterion):
        """Get detailed information about a specific criterion"""
        if disorder in self.criteria and criterion in self.criteria[disorder]:
            return self.criteria[disorder][criterion]
        return None
    
    def get_disorder_criteria_count(self, disorder):
        """Get the minimum criteria required and total criteria for a disorder"""
        if disorder in self.disorders:
            return self.disorders[disorder]
        return None
    
    def get_disorder_info(self, disorder):
        """Get detailed information about a disorder including its criteria."""
        if disorder not in self.disorders:
            return None
            
        return {
            "name": disorder,
            "description": f"Based on your responses, there is a probability of {disorder} Personality Disorder traits.",
            "criteria": [
                {
                    "name": criterion["name"],
                    "description": criterion["description"]
                }
                for criterion in self.criteria[disorder].values()
            ]
        }

# Add this at the end of the file
__all__ = ['ClusterBKnowledgeBase']