"""
Blockchain-based Audit Logging Module
======================================
Implements a tamper-proof audit log for tracking all data access,
model predictions, and system events in RAGChainMed.

Uses SHA-256 hashing to create a blockchain-like chain of audit records,
ensuring data integrity and auditability.

Author: RAGChainMed
Date: May 2026
"""

import hashlib
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import sqlite3
from dataclasses import dataclass, asdict


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class AuditRecord:
    """Represents a single audit log entry"""
    timestamp: str
    user_id: str
    action: str
    data_type: str
    patient_id: Optional[str]
    details: Dict[str, Any]
    status: str  # 'success' or 'failure'
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)


class BlockchainAuditLog:
    """
    Implements a blockchain-like audit log using SHA-256 hashing.
    
    Each record is hashed along with the previous record's hash,
    creating an immutable chain. Any tampering is immediately detectable.
    """
    
    def __init__(self, db_path: str = "audit_logs.db"):
        """
        Initialize the blockchain audit log.
        
        Args:
            db_path: Path to SQLite database for persistent storage
        """
        self.db_path = db_path
        self.chain: List[Dict[str, Any]] = []
        self.previous_hash = "0"
        
        # Initialize database
        self._init_database()
        
        # Load existing chain from database
        self._load_chain_from_db()
    
    # ============================================================
    # DATABASE OPERATIONS
    # ============================================================
    
    def _init_database(self):
        """Initialize SQLite database for audit logs"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_chain (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                block_index INTEGER UNIQUE,
                timestamp TEXT NOT NULL,
                user_id TEXT NOT NULL,
                action TEXT NOT NULL,
                data_type TEXT NOT NULL,
                patient_id TEXT,
                details TEXT NOT NULL,
                status TEXT NOT NULL,
                error_message TEXT,
                current_hash TEXT UNIQUE NOT NULL,
                previous_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_chain_from_db(self):
        """Load existing audit chain from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT block_index, timestamp, user_id, action, data_type, 
                       patient_id, details, status, error_message, 
                       current_hash, previous_hash
                FROM audit_chain
                ORDER BY block_index ASC
            ''')
            
            rows = cursor.fetchall()
            
            for row in rows:
                block = {
                    'index': row[0],
                    'timestamp': row[1],
                    'user_id': row[2],
                    'action': row[3],
                    'data_type': row[4],
                    'patient_id': row[5],
                    'details': json.loads(row[6]),
                    'status': row[7],
                    'error_message': row[8],
                    'current_hash': row[9],
                    'previous_hash': row[10]
                }
                self.chain.append(block)
            
            if self.chain:
                self.previous_hash = self.chain[-1]['current_hash']
            
            conn.close()
        except Exception as e:
            print(f"Warning: Could not load existing chain: {e}")
    
    # ============================================================
    # BLOCKCHAIN OPERATIONS
    # ============================================================
    
    def _create_hash(self, data: str) -> str:
        """
        Create SHA-256 hash of data.
        
        Args:
            data: String data to hash
            
        Returns:
            Hexadecimal hash string
        """
        return hashlib.sha256(data.encode()).hexdigest()
    
    def _compute_block_hash(self, block: Dict[str, Any]) -> str:
        """
        Compute hash for a block combining its data and previous hash.
        
        Args:
            block: Block data dictionary
            
        Returns:
            Block hash
        """
        # Create string representation of block data
        block_data = json.dumps({
            'timestamp': block['timestamp'],
            'user_id': block['user_id'],
            'action': block['action'],
            'data_type': block['data_type'],
            'patient_id': block['patient_id'],
            'details': block['details'],
            'status': block['status'],
            'previous_hash': block['previous_hash']
        }, sort_keys=True)
        
        return self._create_hash(block_data)
    
    def add_record(self, 
                   user_id: str,
                   action: str,
                   data_type: str,
                   details: Dict[str, Any],
                   status: str = "success",
                   patient_id: Optional[str] = None,
                   error_message: Optional[str] = None) -> Dict[str, Any]:
        """
        Add a new audit record to the blockchain.
        
        Args:
            user_id: ID of user performing the action
            action: Type of action (e.g., 'prediction', 'data_access', 'query')
            data_type: Type of data accessed (e.g., 'patient_record', 'clinical_note')
            details: Additional details about the action
            status: 'success' or 'failure'
            patient_id: Associated patient ID if applicable
            error_message: Error message if status is 'failure'
            
        Returns:
            The newly created block
        """
        # Create new block
        block = {
            'index': len(self.chain),
            'timestamp': datetime.utcnow().isoformat(),
            'user_id': user_id,
            'action': action,
            'data_type': data_type,
            'patient_id': patient_id,
            'details': details,
            'status': status,
            'error_message': error_message,
            'previous_hash': self.previous_hash
        }
        
        # Compute hash
        block['current_hash'] = self._compute_block_hash(block)
        
        # Add to chain
        self.chain.append(block)
        self.previous_hash = block['current_hash']
        
        # Persist to database
        self._save_block_to_db(block)
        
        return block
    
    def _save_block_to_db(self, block: Dict[str, Any]):
        """Save a block to the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO audit_chain 
                (block_index, timestamp, user_id, action, data_type, patient_id,
                 details, status, error_message, current_hash, previous_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                block['index'],
                block['timestamp'],
                block['user_id'],
                block['action'],
                block['data_type'],
                block['patient_id'],
                json.dumps(block['details']),
                block['status'],
                block['error_message'],
                block['current_hash'],
                block['previous_hash']
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error saving block to database: {e}")
    
    # ============================================================
    # VERIFICATION & INTEGRITY
    # ============================================================
    
    def verify_chain(self) -> bool:
        """
        Verify the integrity of the entire blockchain.
        
        Returns:
            True if chain is valid (no tampering detected), False otherwise
        """
        if not self.chain:
            return True
        
        for i in range(len(self.chain)):
            block = self.chain[i]
            
            # Verify block's own hash
            computed_hash = self._compute_block_hash(block)
            if computed_hash != block['current_hash']:
                print(f"Block {i}: Hash mismatch!")
                return False
            
            # Verify link to previous block
            if i > 0:
                if block['previous_hash'] != self.chain[i-1]['current_hash']:
                    print(f"Block {i}: Previous hash mismatch!")
                    return False
        
        return True
    
    def get_record_by_index(self, index: int) -> Optional[Dict[str, Any]]:
        """Get a specific record by its index"""
        if 0 <= index < len(self.chain):
            return self.chain[index]
        return None
    
    def get_records_by_user(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all records for a specific user"""
        return [block for block in self.chain if block['user_id'] == user_id]
    
    def get_records_by_patient(self, patient_id: str) -> List[Dict[str, Any]]:
        """Get all records related to a specific patient"""
        return [block for block in self.chain if block['patient_id'] == patient_id]
    
    def get_records_by_action(self, action: str) -> List[Dict[str, Any]]:
        """Get all records of a specific action type"""
        return [block for block in self.chain if block['action'] == action]
    
    def get_recent_records(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get the most recent audit records"""
        return self.chain[-limit:] if len(self.chain) > 0 else []
    
    def generate_audit_report(self, 
                             user_id: Optional[str] = None,
                             patient_id: Optional[str] = None,
                             start_time: Optional[str] = None,
                             end_time: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate an audit report with filtering options.
        
        Args:
            user_id: Filter by specific user
            patient_id: Filter by specific patient
            start_time: Filter by start time (ISO format)
            end_time: Filter by end time (ISO format)
            
        Returns:
            Audit report dictionary
        """
        filtered_records = self.chain
        
        # Apply filters
        if user_id:
            filtered_records = [r for r in filtered_records if r['user_id'] == user_id]
        
        if patient_id:
            filtered_records = [r for r in filtered_records if r['patient_id'] == patient_id]
        
        if start_time:
            filtered_records = [r for r in filtered_records if r['timestamp'] >= start_time]
        
        if end_time:
            filtered_records = [r for r in filtered_records if r['timestamp'] <= end_time]
        
        # Generate statistics
        report = {
            'generated_at': datetime.utcnow().isoformat(),
            'total_records': len(filtered_records),
            'chain_integrity_verified': self.verify_chain(),
            'action_summary': {},
            'status_summary': {},
            'records': filtered_records
        }
        
        # Summarize by action
        for action in set(r['action'] for r in filtered_records):
            count = len([r for r in filtered_records if r['action'] == action])
            report['action_summary'][action] = count
        
        # Summarize by status
        for status in set(r['status'] for r in filtered_records):
            count = len([r for r in filtered_records if r['status'] == status])
            report['status_summary'][status] = count
        
        return report
    
    def export_to_json(self, file_path: str):
        """Export the entire chain to JSON file"""
        export_data = {
            'chain_length': len(self.chain),
            'integrity_verified': self.verify_chain(),
            'exported_at': datetime.utcnow().isoformat(),
            'blocks': self.chain
        }
        
        with open(file_path, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def __len__(self) -> int:
        """Return the length of the chain"""
        return len(self.chain)
    
    def __repr__(self) -> str:
        return f"BlockchainAuditLog(length={len(self.chain)}, integrity_verified={self.verify_chain()})"
