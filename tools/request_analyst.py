#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
import os
import sys
import argparse
from datetime import datetime
from typing import List, Tuple, Optional
import ast

# Maximum file size to process (in bytes) - 100MB
MAX_FILE_SIZE = 100 * 1024 * 1024
# Maximum line length to process
MAX_LINE_LENGTH = 1000000

def convert_to_json(data: str) -> str:
    """
    Convert Python dict string to valid JSON string
    """
    try:
        # First handle enum types
        data = re.sub(r'<MessageRole\.[A-Z]+:\s*\'([^\']+)\'>', r"'\1'", data)
        data = re.sub(r'<DataSourceType\.[A-Z]+:\s*(\d+)>', r'\1', data)
        
        # Use ast.literal_eval to safely evaluate the Python string
        python_dict = ast.literal_eval(data)
        
        # Convert Python dict to JSON string
        return json.dumps(python_dict)
    except Exception as e:
        print(f"Warning: Error converting to JSON: {str(e)}")
        return data

def check_file_size(file_path: str) -> int:
    """
    Check if file size is within acceptable limits
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Log file not found: {file_path}")
        
    file_size = os.path.getsize(file_path)
    if file_size > MAX_FILE_SIZE:
        raise ValueError(f"File size ({file_size / 1024 / 1024:.2f}MB) exceeds maximum allowed size ({MAX_FILE_SIZE / 1024 / 1024:.2f}MB)")
    if file_size == 0:
        raise ValueError("Log file is empty")
    return file_size

def extract_segments(content: str) -> List[Tuple[str, Optional[str], str]]:
    """
    Extract text, skill and answer segments from content
    Returns: List of tuples (type_name, subtype, content)
    """
    if not content:
        return [('text', None, '')]
    
    segments = []
    pattern = r'<([^>]+)>(.*?)</\1>'
    pos = 0
    
    try:
        for match in re.finditer(pattern, content, re.DOTALL):
            start, end = match.span()
            tag_name = match.group(1)
            tag_content = match.group(2)
            
            # Add text before tag
            if start > pos:
                text = content[pos:start].strip()
                if text:
                    segments.append(('text', None, text))
            
            # Add tag content
            if tag_name == 'answer':
                segments.append(('answer', None, tag_content.strip()))
            else:
                segments.append(('skill', tag_name, tag_content.strip()))
            
            pos = end
        
        # Add remaining text
        if pos < len(content):
            text = content[pos:].strip()
            if text:
                segments.append(('text', None, text))
                
        return segments or [('text', None, content)]
    except Exception as e:
        print(f"Warning: Error extracting segments: {str(e)}")
        return [('text', None, content)]

def format_table_row(type_name: str, subtype: Optional[str], content: str) -> str:
    """
    Format table row with proper alignment
    """
    try:
        content_preview = content.replace('\n', '\\n')
        
        return f"{type_name:<10}\t{subtype if subtype else '':<16}\t{len(content):<8}\t{content_preview}"
    except Exception as e:
        print(f"Warning: Error formatting table row: {str(e)}")
        return f"error\t\t0\tError formatting content: {str(e)}"

def process_request(request_data: str, request_num: int) -> List[str]:
    """
    Process single request data
    """
    results = []
    results.append(f"\n{'=' * 50}")
    results.append(f"Request #{request_num}")

    try:
        if len(request_data) > MAX_LINE_LENGTH:
            raise ValueError(f"Request data too long: {len(request_data)} chars")
            
        # Extract the actual request data from the log line
        request_data = request_data.strip()
        if not request_data:
            raise ValueError("Empty request data")
            
        # Convert Python dict format to JSON
        json_data = convert_to_json(request_data)
        request_log = json.loads(json_data)

        if not isinstance(request_log, dict):
            raise ValueError(f"Invalid request format: expected dict, got {type(request_log)}")
            
        if 'json_data' not in request_log:
            raise ValueError("Missing 'json_data' field in request")
            
        if not isinstance(request_log['json_data'], dict):
            raise ValueError(f"Invalid json_data format: expected dict, got {type(request_log['json_data'])}")
            
        if 'messages' not in request_log['json_data']:
            raise ValueError("Missing 'messages' field in json_data")
            
        messages = request_log['json_data']['messages']
        
        if not isinstance(messages, list):
            raise ValueError(f"Invalid messages format: expected list, got {type(messages)}")
            
        for msg_idx, message in enumerate(messages, 1):
            results.append(f"\n{'-' * 30}")
            results.append(f"Message #{msg_idx}")
            
            if not isinstance(message, dict):
                results.append(f"Error: Invalid message format at index {msg_idx}")
                continue
                
            role = message.get('role', 'unknown')
            content = message.get('content', '')
            
            results.append(f"Role: {role}")
            
            try:
                segments = extract_segments(content)
                for segment in segments:
                    results.append(format_table_row(*segment))
            except Exception as e:
                results.append(f"Error parsing message content: {str(e)}")
                results.append(format_table_row('text', None, content))
            
            results.append(f"{'-' * 30}")
            
    except json.JSONDecodeError as e:
        results.append(f"Error parsing JSON at position {e.pos}: {str(e)}")
        results.append(f"Raw data preview: {request_data[:200]}...")
        results.append(f"Converted JSON preview: {json_data[:200]}...")
    except Exception as e:
        results.append(f"Error processing request: {str(e)}")
        
    results.append(f"{'=' * 50}\n")
    return results

def analyze_log_file(file_path: str) -> str:
    """
    Analyze log file and return formatted results
    """
    results = []
    request_count = 0
    error_count = 0
    
    try:
        file_size = check_file_size(file_path)
        print(f"Processing file of size: {file_size / 1024 / 1024:.2f}MB")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            line_count = 0
            
            for line_num, line in enumerate(f, 1):
                try:
                    line_count += 1
                    if line_count % 1000 == 0:
                        print(f"Processed {line_count} lines...")
                    
                    # Skip lines without Request options
                    if 'Request options:' not in line:
                        continue
                        
                    # Extract JSON part after Request options:
                    json_part = line[line.find('Request options:') + len('Request options:'):].strip()
                    
                    # Only process if the line contains complete JSON (starts with { and ends with })
                    if json_part.startswith('{') and json_part.endswith('}'):
                        request_count += 1
                        results.extend(process_request(json_part, request_count))
                
                except Exception as e:
                    error_count += 1
                    print(f"Warning: Error at line {line_num}: {str(e)}")
                    if error_count > 100:
                        raise Exception("Too many errors encountered during processing")
                    continue
        
        if request_count == 0:
            results.append("No valid requests found in the log file")
        else:
            summary = [
                f"\nAnalysis Summary:",
                f"Total lines processed: {line_count}",
                f"Total requests found: {request_count}",
                f"Total errors encountered: {error_count}",
                f"Success rate: {((request_count - error_count) / request_count * 100):.2f}%\n"
            ]
            results = summary + results
        
    except Exception as e:
        results.append(f"Fatal error processing log file: {str(e)}")
    
    return '\n'.join(results)

def main():
    parser = argparse.ArgumentParser(description='Analyze request logs')
    parser.add_argument('log_file', help='Path to the log file to analyze')
    args = parser.parse_args()
    
    try:
        results = analyze_log_file(args.log_file)
        
        # Generate output filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        output_file = f"results/request_analysis_{timestamp}.txt"
        
        # Ensure results directory exists
        os.makedirs('results', exist_ok=True)
        
        # Write results to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(results)
            
        print(f"Analysis complete. Results saved to {output_file}")
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 