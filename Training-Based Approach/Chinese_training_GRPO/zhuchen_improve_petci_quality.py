#!/usr/bin/env python3
"""
Improved PETCI Dataset Quality Filter

This script applies additional quality filters to remove low-quality entries:
1. Remove entries where literal_translation is too similar to true_meaning
2. Remove entries with poor literal translations (too short, meaningless)
3. Remove entries where literal translation is just a word-for-word translation
4. Keep only entries with meaningful contrast between figurative and literal meanings
"""

import csv
import re
from typing import List, Dict, Any


def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate simple word overlap similarity between two texts"""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0


def is_meaningless_literal(translation: str) -> bool:
    """Check if a literal translation is meaningless or too simple"""
    translation = translation.lower().strip()
    
    # Too short (less than 3 words)
    if len(translation.split()) < 3:
        return True
    
    # Just a single word or very basic phrase
    meaningless_patterns = [
        r'^a\s+\w+$',  # "a word"
        r'^the\s+\w+$',  # "the word"
        r'^\w+\s+and\s+\w+$',  # "word and word"
        r'^\w+\s+of\s+\w+$',  # "word of word"
        r'^one\s+\w+$',  # "one word"
        r'^\w+\s+one$',  # "word one"
    ]
    
    for pattern in meaningless_patterns:
        if re.match(pattern, translation):
            return True
    
    return False


def is_word_for_word_translation(literal: str, figurative: str) -> bool:
    """Check if literal translation is just a word-for-word version of figurative"""
    literal_words = literal.lower().split()
    figurative_words = figurative.lower().split()
    
    # If literal has more than 80% word overlap with figurative, it's likely word-for-word
    overlap = len(set(literal_words).intersection(set(figurative_words)))
    total_unique = len(set(literal_words).union(set(figurative_words)))
    
    if total_unique == 0:
        return False
    
    overlap_ratio = overlap / total_unique
    return overlap_ratio > 0.8


def has_meaningful_contrast(figurative: str, literal: str) -> bool:
    """Check if there's a meaningful contrast between figurative and literal meanings"""
    
    # If they're too similar, no meaningful contrast
    if calculate_similarity(figurative, literal) > 0.7:
        return False
    
    # If literal is meaningless, no meaningful contrast
    if is_meaningless_literal(literal):
        return False
    
    # If literal is just word-for-word, no meaningful contrast
    if is_word_for_word_translation(literal, figurative):
        return False
    
    return True


def improve_petci_quality(input_file: str, output_file: str) -> None:
    """Apply quality improvements to the PETCI dataset"""
    
    print(f"Loading data from {input_file}...")
    
    # Load the existing CSV
    entries = []
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        entries = list(reader)
    
    print(f"Loaded {len(entries)} entries")
    
    # Apply quality filters
    print("Applying quality improvements...")
    improved_entries = []
    removed_count = 0
    
    for entry in entries:
        figurative = entry['true_meaning']
        literal = entry['literal_translation']
        
        # Check if entry has meaningful contrast
        if has_meaningful_contrast(figurative, literal):
            improved_entries.append(entry)
        else:
            removed_count += 1
            if removed_count <= 10:  # Show first 10 removed entries as examples
                print(f"Removed: {entry['src']} -> {figurative} | {literal}")
    
    print(f"Removed {removed_count} low-quality entries")
    print(f"Kept {len(improved_entries)} high-quality entries")
    
    # Save improved dataset
    print(f"Creating improved CSV file: {output_file}")
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['src', 'true_meaning', 'literal_translation'])
        writer.writeheader()
        writer.writerows(improved_entries)
    
    print(f"Successfully created {output_file} with {len(improved_entries)} entries")
    
    # Print some sample high-quality entries
    print("\nSample high-quality entries:")
    for i, entry in enumerate(improved_entries[:10]):
        print(f"{i+1}. {entry['src']} -> {entry['true_meaning']} | {entry['literal_translation']}")


def main():
    """Main function"""
    input_file = "/u/zshao7/homework/petci_chinese_english.csv"
    output_file = "/u/zshao7/homework/petci_chinese_english_improved.csv"
    
    try:
        improve_petci_quality(input_file, output_file)
    except Exception as e:
        print(f"Error improving data quality: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
