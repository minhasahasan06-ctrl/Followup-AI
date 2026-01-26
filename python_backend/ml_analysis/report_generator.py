"""
Report Generator for Research Center

Implements LLM API integration to generate research narratives
including Abstract, Methods, Results, and Discussion/Limitations.
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import openai

@dataclass
class ReportConfig:
    """Configuration for report generation"""
    model: str = 'gpt-4o'
    temperature: float = 0.7
    max_tokens: int = 4000
    include_abstract: bool = True
    include_methods: bool = True
    include_results: bool = True
    include_discussion: bool = True
    include_limitations: bool = True
    style: str = 'academic'  # 'academic', 'clinical', 'summary'

class ReportGenerator:
    """
    AI-powered research report generator.
    
    Features:
    - Structured narrative generation (Abstract, Methods, Results, Discussion)
    - Multiple output styles (academic, clinical, summary)
    - Table and figure caption generation
    - Statistical result interpretation
    - Limitation analysis
    - HIPAA-compliant text generation
    """
    
    def __init__(self, config: Optional[ReportConfig] = None):
        self.config = config or ReportConfig()
        self.client = openai.OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
    
    def generate_full_report(
        self,
        study_info: Dict[str, Any],
        analysis_results: Dict[str, Any],
        cohort_description: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a complete research report.
        
        Args:
            study_info: Study metadata (title, objective, design)
            analysis_results: Analysis output (descriptive, risk, survival, causal)
            cohort_description: Description of study cohort
        
        Returns:
            Dictionary with all report sections
        """
        report = {
            'generated_at': datetime.utcnow().isoformat(),
            'study_title': study_info.get('title', 'Untitled Study'),
            'sections': {}
        }
        
        if self.config.include_abstract:
            report['sections']['abstract'] = self.generate_abstract(
                study_info, analysis_results
            )
        
        if self.config.include_methods:
            report['sections']['methods'] = self.generate_methods(
                study_info, analysis_results, cohort_description
            )
        
        if self.config.include_results:
            report['sections']['results'] = self.generate_results(
                analysis_results
            )
        
        if self.config.include_discussion:
            report['sections']['discussion'] = self.generate_discussion(
                study_info, analysis_results
            )
        
        if self.config.include_limitations:
            report['sections']['limitations'] = self.generate_limitations(
                analysis_results
            )
        
        return report
    
    def generate_abstract(
        self,
        study_info: Dict[str, Any],
        analysis_results: Dict[str, Any]
    ) -> str:
        """Generate structured abstract"""
        prompt = f"""
        Generate a structured scientific abstract for the following research study.
        
        Study Information:
        - Title: {study_info.get('title', 'N/A')}
        - Objective: {study_info.get('objective', 'N/A')}
        - Design: {study_info.get('design', 'N/A')}
        
        Key Results:
        {self._format_key_results(analysis_results)}
        
        Generate an abstract with the following sections:
        - Background (2-3 sentences)
        - Objective (1-2 sentences)
        - Methods (2-3 sentences)
        - Results (3-4 sentences with key statistics)
        - Conclusion (1-2 sentences)
        
        Style: {self.config.style}
        Format as a single flowing paragraph with section labels in bold.
        Do not include any patient identifiers or PHI.
        """
        
        return self._generate_text(prompt)
    
    def generate_methods(
        self,
        study_info: Dict[str, Any],
        analysis_results: Dict[str, Any],
        cohort_description: Optional[str] = None
    ) -> str:
        """Generate methods section"""
        methods_info = {
            'study_design': study_info.get('design', 'observational cohort study'),
            'population': cohort_description or 'Adult patients from the study cohort',
            'data_sources': 'Electronic health records, daily symptom tracking, clinical assessments'
        }
        
        analysis_types = []
        if 'descriptive' in analysis_results:
            analysis_types.append('Descriptive statistics')
        if 'risk_prediction' in analysis_results:
            analysis_types.append('Risk prediction modeling')
        if 'survival' in analysis_results:
            analysis_types.append('Survival analysis')
        if 'causal' in analysis_results:
            analysis_types.append('Causal inference analysis')
        
        prompt = f"""
        Generate a Methods section for a research paper.
        
        Study Design: {methods_info['study_design']}
        Study Population: {methods_info['population']}
        Data Sources: {methods_info['data_sources']}
        
        Analyses Performed:
        {', '.join(analysis_types) if analysis_types else 'Descriptive analysis'}
        
        Include subsections for:
        1. Study Design and Setting
        2. Study Population (inclusion/exclusion criteria)
        3. Data Collection
        4. Statistical Analysis
        5. Ethical Considerations
        
        Style: {self.config.style}
        Use passive voice and past tense as appropriate for scientific writing.
        Mention that all analyses were conducted with HIPAA compliance.
        """
        
        return self._generate_text(prompt)
    
    def generate_results(
        self,
        analysis_results: Dict[str, Any]
    ) -> str:
        """Generate results section with statistical findings"""
        results_text = []
        
        if 'descriptive' in analysis_results:
            desc = analysis_results['descriptive']
            if 'table1' in desc:
                results_text.append(self._describe_table1(desc['table1']))
        
        if 'risk_prediction' in analysis_results:
            risk = analysis_results['risk_prediction']
            results_text.append(self._describe_risk_model(risk))
        
        if 'survival' in analysis_results:
            surv = analysis_results['survival']
            results_text.append(self._describe_survival(surv))
        
        if 'causal' in analysis_results:
            causal = analysis_results['causal']
            results_text.append(self._describe_causal(causal))
        
        results_summary = "\n\n".join(results_text) if results_text else "No analysis results available."
        
        prompt = f"""
        Generate a well-written Results section based on the following analysis outputs:
        
        {results_summary}
        
        Structure the results logically:
        1. Start with population characteristics
        2. Present main findings with statistics
        3. Include secondary analyses
        4. Report any sensitivity analyses
        
        Style: {self.config.style}
        Include exact statistics with confidence intervals and p-values where available.
        Use proper statistical reporting conventions (e.g., OR 2.5, 95% CI 1.8-3.5, p<0.001).
        """
        
        return self._generate_text(prompt)
    
    def generate_discussion(
        self,
        study_info: Dict[str, Any],
        analysis_results: Dict[str, Any]
    ) -> str:
        """Generate discussion section"""
        key_findings = self._extract_key_findings(analysis_results)
        
        prompt = f"""
        Generate a Discussion section for a research paper.
        
        Study Objective: {study_info.get('objective', 'N/A')}
        
        Key Findings:
        {key_findings}
        
        Include:
        1. Summary of main findings (1 paragraph)
        2. Comparison with existing literature (2-3 paragraphs)
        3. Clinical implications (1-2 paragraphs)
        4. Strengths of the study (1 paragraph)
        5. Future research directions (1 paragraph)
        
        Style: {self.config.style}
        Be balanced and objective. Acknowledge both confirmatory and novel findings.
        Do not include limitations here (they will be in a separate section).
        """
        
        return self._generate_text(prompt)
    
    def generate_limitations(
        self,
        analysis_results: Dict[str, Any]
    ) -> str:
        """Generate limitations section"""
        potential_limitations = []
        
        if 'descriptive' in analysis_results:
            desc = analysis_results['descriptive']
            missing = desc.get('missing_data', {})
            if missing.get('complete_case_pct', 100) < 80:
                potential_limitations.append('Significant missing data that may introduce bias')
        
        if 'risk_prediction' in analysis_results:
            risk = analysis_results['risk_prediction']
            if risk.get('n_samples', 0) < 200:
                potential_limitations.append('Limited sample size affecting model generalizability')
            if risk.get('metrics', {}).get('auroc', 0) < 0.7:
                potential_limitations.append('Moderate discrimination performance of predictive model')
        
        if 'causal' in analysis_results:
            potential_limitations.extend([
                'Potential unmeasured confounding in observational design',
                'Selection bias from inclusion criteria'
            ])
        
        prompt = f"""
        Generate a Limitations section for a research paper.
        
        Identified limitations:
        {json.dumps(potential_limitations, indent=2) if potential_limitations else 'Standard observational study limitations'}
        
        Include:
        1. Study design limitations
        2. Data quality and measurement issues
        3. Generalizability concerns
        4. Potential biases (selection, information, confounding)
        5. How limitations might affect interpretation
        
        Style: {self.config.style}
        Be honest but not overly negative. Suggest how limitations were addressed where applicable.
        Keep this section focused and around 200-300 words.
        """
        
        return self._generate_text(prompt)
    
    def generate_table_caption(
        self,
        table_type: str,
        table_data: Dict[str, Any]
    ) -> str:
        """Generate caption for a table"""
        prompt = f"""
        Generate a concise table caption for a research paper.
        
        Table Type: {table_type}
        Content Summary: {json.dumps(table_data.get('summary', {}), indent=2)[:500]}
        
        The caption should:
        - Start with "Table X." (leave X as placeholder)
        - Describe what the table shows
        - Include any abbreviations used
        - Be 1-2 sentences
        
        Style: {self.config.style}
        """
        
        return self._generate_text(prompt, max_tokens=150)
    
    def generate_figure_caption(
        self,
        figure_type: str,
        figure_data: Dict[str, Any]
    ) -> str:
        """Generate caption for a figure"""
        prompt = f"""
        Generate a figure caption for a research paper.
        
        Figure Type: {figure_type}
        Content: {json.dumps(figure_data.get('description', {}), indent=2)[:500]}
        
        The caption should:
        - Start with "Figure X." (leave X as placeholder)
        - Describe what the figure shows
        - Explain any abbreviations or symbols
        - Note sample sizes if relevant
        - Be 2-3 sentences
        
        Style: {self.config.style}
        """
        
        return self._generate_text(prompt, max_tokens=200)
    
    def _generate_text(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """Generate text using LLM API"""
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert medical researcher and scientific writer. Generate clear, accurate, and well-structured research content. Always maintain HIPAA compliance - never include patient identifiers or PHI."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.config.temperature,
                max_tokens=max_tokens or self.config.max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"[Error generating content: {str(e)}]"
    
    def _format_key_results(self, analysis_results: Dict[str, Any]) -> str:
        """Format key results for prompts"""
        results = []
        
        if 'descriptive' in analysis_results:
            desc = analysis_results['descriptive']
            if 'table1' in desc:
                results.append(f"Sample size: {desc['table1'].get('n_total', 'N/A')}")
        
        if 'risk_prediction' in analysis_results:
            risk = analysis_results['risk_prediction']
            metrics = risk.get('metrics', {})
            results.append(f"Risk model AUROC: {metrics.get('auroc', 'N/A'):.3f}" if metrics.get('auroc') else "")
            results.append(f"Event rate: {risk.get('event_rate', 0):.1%}" if risk.get('event_rate') else "")
        
        if 'survival' in analysis_results:
            surv = analysis_results['survival']
            km = surv.get('kaplan_meier', {})
            if km.get('median_survival'):
                results.append(f"Median survival: {km['median_survival']:.1f} {km.get('time_unit', 'days')}")
        
        if 'causal' in analysis_results:
            causal = analysis_results['causal']
            ate = causal.get('ate', {})
            if ate.get('ate') is not None:
                results.append(f"Average treatment effect: {ate['ate']:.3f} (95% CI: {ate.get('ci_lower', 'N/A'):.3f} to {ate.get('ci_upper', 'N/A'):.3f})")
        
        return "\n".join([r for r in results if r])
    
    def _describe_table1(self, table1: Dict[str, Any]) -> str:
        """Generate text description of Table 1"""
        n_total = table1.get('n_total', 0)
        variables = table1.get('variables', [])
        
        continuous = [v for v in variables if v.get('type') == 'continuous']
        categorical = [v for v in variables if v.get('type') == 'categorical']
        
        return f"""
        Baseline Characteristics:
        - Total sample: {n_total} patients
        - Continuous variables analyzed: {len(continuous)}
        - Categorical variables analyzed: {len(categorical)}
        """
    
    def _describe_risk_model(self, risk: Dict[str, Any]) -> str:
        """Generate text description of risk model results"""
        metrics = risk.get('metrics', {})
        
        return f"""
        Risk Prediction Model ({risk.get('model_type', 'N/A')}):
        - Sample size: {risk.get('n_samples', 'N/A')} (events: {risk.get('n_events', 'N/A')})
        - AUROC: {metrics.get('auroc', 0):.3f}
        - AUPRC: {metrics.get('auprc', 0):.3f}
        - Brier score: {metrics.get('brier_score', 0):.3f}
        - Sensitivity: {metrics.get('sensitivity', 0):.1%}
        - Specificity: {metrics.get('specificity', 0):.1%}
        """
    
    def _describe_survival(self, survival: Dict[str, Any]) -> str:
        """Generate text description of survival analysis"""
        km = survival.get('kaplan_meier', {})
        
        return f"""
        Survival Analysis:
        - Total patients: {km.get('n_total', 'N/A')}
        - Events: {km.get('n_events', 'N/A')}
        - Censored: {km.get('n_censored', 'N/A')}
        - Median survival: {km.get('median_survival', 'N/A')} {km.get('time_unit', 'days')}
        """
    
    def _describe_causal(self, causal: Dict[str, Any]) -> str:
        """Generate text description of causal analysis"""
        ate = causal.get('ate', {})
        balance = causal.get('covariate_balance', {})
        
        return f"""
        Causal Analysis ({ate.get('method', 'N/A')}):
        - Average Treatment Effect: {ate.get('ate', 'N/A')}
        - 95% CI: ({ate.get('ci_lower', 'N/A')}, {ate.get('ci_upper', 'N/A')})
        - P-value: {ate.get('p_value', 'N/A')}
        - SMD after weighting: {balance.get('overall_smd_after', 'N/A')}
        """
    
    def _extract_key_findings(self, analysis_results: Dict[str, Any]) -> str:
        """Extract key findings for discussion"""
        findings = []
        
        if 'risk_prediction' in analysis_results:
            risk = analysis_results['risk_prediction']
            if risk.get('metrics', {}).get('auroc', 0) > 0.7:
                findings.append(f"The risk model achieved good discrimination (AUROC > 0.7)")
            
            top_features = risk.get('feature_importance', [])[:3]
            if top_features:
                features = [f['feature'] for f in top_features]
                findings.append(f"Key predictors included: {', '.join(features)}")
        
        if 'causal' in analysis_results:
            ate = analysis_results['causal'].get('ate', {})
            if ate.get('significant'):
                findings.append(f"Treatment effect was statistically significant (ATE: {ate.get('ate', 0):.3f})")
        
        return "\n".join(findings) if findings else "Main findings to be discussed"


def generate_research_report(
    study_info: Dict[str, Any],
    analysis_results: Dict[str, Any],
    style: str = 'academic'
) -> Dict[str, Any]:
    """
    Convenience function to generate a research report.
    
    Args:
        study_info: Study metadata
        analysis_results: Analysis outputs
        style: 'academic', 'clinical', or 'summary'
    
    Returns:
        Complete report with all sections
    """
    config = ReportConfig(style=style)
    generator = ReportGenerator(config)
    
    return generator.generate_full_report(study_info, analysis_results)
