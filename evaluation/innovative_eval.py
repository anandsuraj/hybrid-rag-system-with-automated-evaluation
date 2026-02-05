"""
Innovative Evaluation Features
Includes ablation studies and error analysis
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from typing import Dict, List
import config


def run_ablation_study(retriever, generator) -> Dict:
    """
    Run ablation study comparing dense-only, sparse-only, and hybrid retrieval.
    
    Args:
        retriever: HybridRetriever instance
        generator: ResponseGenerator instance
    
    Returns:
        Dictionary with ablation study results
    """
    print("\nRunning ablation study...")
    
    # Load questions
    with open(config.QUESTIONS_FILE, 'r') as f:
        questions_data = json.load(f)
    
    # Sample 20 questions for ablation (to save time)
    import random
    sample_questions = random.sample(questions_data, min(20, len(questions_data)))
    
    results = {
        'dense_only': {'correct': 0, 'total': 0},
        'sparse_only': {'correct': 0, 'total': 0},
        'hybrid': {'correct': 0, 'total': 0}
    }
    
    for i, q in enumerate(sample_questions):
        print(f"  Testing question {i+1}/{len(sample_questions)}...", end='\r')
        
        question = q['question']
        source_url = q['source_url']
        
        # Parse source URLs (handle comma-separated strings for comparative questions)
        if isinstance(source_url, list):
            source_urls = source_url
        elif ',' in source_url:
            source_urls = [url.strip() for url in source_url.split(',')]
        else:
            source_urls = [source_url]
        
        # Test dense-only
        try:
            dense_results = retriever.dense_retriever.search(question, top_k=5)
            dense_urls = [chunk['url'] for chunk, _ in dense_results]
            results['dense_only']['correct'] += any(url in dense_urls for url in source_urls)
            results['dense_only']['total'] += 1
        except Exception as e:
            pass
        
        # Test sparse-only
        try:
            sparse_results = retriever.sparse_retriever.search(question, top_k=5)
            sparse_urls = [chunk['url'] for chunk, _ in sparse_results]
            results['sparse_only']['correct'] += any(url in sparse_urls for url in source_urls)
            results['sparse_only']['total'] += 1
        except Exception as e:
            pass
        
        # Test hybrid
        try:
            hybrid_results, _ = retriever.search(question, final_top_n=5)
            hybrid_urls = [chunk['url'] for chunk, _ in hybrid_results]
            results['hybrid']['correct'] += any(url in hybrid_urls for url in source_urls)
            results['hybrid']['total'] += 1
        except Exception as e:
            pass
    
    print()  # New line after progress
    
    # Calculate accuracy
    for method in results:
        if results[method]['total'] > 0:
            results[method]['accuracy'] = results[method]['correct'] / results[method]['total']
        else:
            results[method]['accuracy'] = 0.0
    
    print(f"  Dense-only accuracy: {results['dense_only']['accuracy']:.3f}")
    print(f"  Sparse-only accuracy: {results['sparse_only']['accuracy']:.3f}")
    print(f"  Hybrid accuracy: {results['hybrid']['accuracy']:.3f}")
    
    return results


def analyze_errors(results: Dict) -> Dict:
    """
    Analyze errors by question type and failure mode.
    
    Args:
        results: Evaluation results dictionary
    
    Returns:
        Dictionary with error analysis
    """
    print("\nAnalyzing errors by question type...")
    
    error_analysis = {
        'by_type': {
            'factual': {'total': 0, 'failed': 0},
            'comparative': {'total': 0, 'failed': 0},
            'inferential': {'total': 0, 'failed': 0},
            'multi_hop': {'total': 0, 'failed': 0}
        },
        'failure_modes': {
            'retrieval_failure': 0,
            'generation_failure': 0,
            'both_failure': 0
        }
    }
    
    # Analyze per-question results
    if 'per_question_results' in results:
        for q_result in results['per_question_results']:
            q_type = q_result.get('question_type', 'unknown')
            
            if q_type in error_analysis['by_type']:
                error_analysis['by_type'][q_type]['total'] += 1
                
                # Check if retrieval failed (MRR = 0)
                retrieval_failed = q_result.get('mrr', 0) == 0
                
                # Check if generation failed (ROUGE-L F1 < 0.2)
                generation_failed = q_result.get('rouge_l_f1', 0) < 0.2
                
                if retrieval_failed and generation_failed:
                    error_analysis['by_type'][q_type]['failed'] += 1
                    error_analysis['failure_modes']['both_failure'] += 1
                elif retrieval_failed:
                    error_analysis['by_type'][q_type]['failed'] += 1
                    error_analysis['failure_modes']['retrieval_failure'] += 1
                elif generation_failed:
                    error_analysis['by_type'][q_type]['failed'] += 1
                    error_analysis['failure_modes']['generation_failure'] += 1
    
    # Calculate failure rates
    for q_type in error_analysis['by_type']:
        total = error_analysis['by_type'][q_type]['total']
        if total > 0:
            failed = error_analysis['by_type'][q_type]['failed']
            error_analysis['by_type'][q_type]['failure_rate'] = failed / total
        else:
            error_analysis['by_type'][q_type]['failure_rate'] = 0.0
    
    print("  Error rates by question type:")
    for q_type, stats in error_analysis['by_type'].items():
        if stats['total'] > 0:
            print(f"    {q_type}: {stats['failure_rate']:.1%} ({stats['failed']}/{stats['total']})")
    
    return error_analysis


if __name__ == "__main__":
    print("This module should be imported, not run directly.")
    print("Use run_evaluation.py to run the complete evaluation pipeline.")
