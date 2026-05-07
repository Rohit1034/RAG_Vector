"""
Access Control and Permissions Module
=====================================
Manages role-based access control (RBAC) for RAGChainMed system.
Tracks user roles and enforces permission policies.

Author: RAGChainMed
Date: May 2026
"""

from typing import Dict, List, Set, Optional
from enum import Enum
from datetime import datetime, timedelta


class UserRole(Enum):
    """Enumeration of user roles in the system"""
    ADMIN = "admin"
    DOCTOR = "doctor"
    NURSE = "nurse"
    RESEARCHER = "researcher"
    AUDIT_OFFICER = "audit_officer"
    SYSTEM = "system"


class Permission(Enum):
    """Enumeration of permissions"""
    # Patient data access
    VIEW_PATIENT_DATA = "view_patient_data"
    EDIT_PATIENT_DATA = "edit_patient_data"
    DELETE_PATIENT_DATA = "delete_patient_data"
    
    # Model predictions
    REQUEST_PREDICTION = "request_prediction"
    VIEW_PREDICTION = "view_prediction"
    RETRAIN_MODEL = "retrain_model"
    
    # System management
    MANAGE_USERS = "manage_users"
    VIEW_AUDIT_LOGS = "view_audit_logs"
    EXPORT_DATA = "export_data"
    
    # Knowledge base
    QUERY_KNOWLEDGE_BASE = "query_knowledge_base"
    ADD_KNOWLEDGE = "add_knowledge"


# Define role-based permissions mapping
ROLE_PERMISSIONS = {
    UserRole.ADMIN: {
        Permission.VIEW_PATIENT_DATA,
        Permission.EDIT_PATIENT_DATA,
        Permission.DELETE_PATIENT_DATA,
        Permission.REQUEST_PREDICTION,
        Permission.VIEW_PREDICTION,
        Permission.RETRAIN_MODEL,
        Permission.MANAGE_USERS,
        Permission.VIEW_AUDIT_LOGS,
        Permission.EXPORT_DATA,
        Permission.QUERY_KNOWLEDGE_BASE,
        Permission.ADD_KNOWLEDGE,
    },
    UserRole.DOCTOR: {
        Permission.VIEW_PATIENT_DATA,
        Permission.EDIT_PATIENT_DATA,
        Permission.REQUEST_PREDICTION,
        Permission.VIEW_PREDICTION,
        Permission.QUERY_KNOWLEDGE_BASE,
    },
    UserRole.NURSE: {
        Permission.VIEW_PATIENT_DATA,
        Permission.EDIT_PATIENT_DATA,
        Permission.QUERY_KNOWLEDGE_BASE,
    },
    UserRole.RESEARCHER: {
        Permission.VIEW_PATIENT_DATA,
        Permission.REQUEST_PREDICTION,
        Permission.VIEW_PREDICTION,
        Permission.QUERY_KNOWLEDGE_BASE,
        Permission.EXPORT_DATA,
        Permission.RETRAIN_MODEL,
    },
    UserRole.AUDIT_OFFICER: {
        Permission.VIEW_AUDIT_LOGS,
        Permission.VIEW_PATIENT_DATA,
        Permission.VIEW_PREDICTION,
    },
    UserRole.SYSTEM: {
        Permission.VIEW_PATIENT_DATA,
        Permission.REQUEST_PREDICTION,
        Permission.VIEW_PREDICTION,
        Permission.QUERY_KNOWLEDGE_BASE,
    },
}


class User:
    """Represents a user in the system"""
    
    def __init__(self, user_id: str, username: str, role: UserRole, 
                 department: str = ""):
        self.user_id = user_id
        self.username = username
        self.role = role
        self.department = department
        self.created_at = datetime.utcnow()
        self.last_login = None
        self.is_active = True
    
    def has_permission(self, permission: Permission) -> bool:
        """Check if user has a specific permission"""
        return permission in ROLE_PERMISSIONS.get(self.role, set())
    
    def has_any_permission(self, permissions: List[Permission]) -> bool:
        """Check if user has any of the listed permissions"""
        user_perms = ROLE_PERMISSIONS.get(self.role, set())
        return any(p in user_perms for p in permissions)
    
    def has_all_permissions(self, permissions: List[Permission]) -> bool:
        """Check if user has all of the listed permissions"""
        user_perms = ROLE_PERMISSIONS.get(self.role, set())
        return all(p in user_perms for p in permissions)
    
    def get_permissions(self) -> Set[Permission]:
        """Get all permissions for this user"""
        return ROLE_PERMISSIONS.get(self.role, set())
    
    def to_dict(self) -> Dict:
        """Convert user to dictionary"""
        return {
            'user_id': self.user_id,
            'username': self.username,
            'role': self.role.value,
            'department': self.department,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat(),
            'last_login': self.last_login.isoformat() if self.last_login else None
        }


class AccessControlManager:
    """Manages user access control and permissions"""
    
    def __init__(self):
        self.users: Dict[str, User] = {}
        # Create default system user
        self._create_system_user()
    
    def _create_system_user(self):
        """Create a system user for automated operations"""
        system_user = User("system", "RAGChainMed System", UserRole.SYSTEM)
        self.users["system"] = system_user
    
    def add_user(self, user: User) -> bool:
        """Add a new user to the system"""
        if user.user_id in self.users:
            return False
        self.users[user.user_id] = user
        return True
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        return self.users.get(user_id)
    
    def update_user_role(self, user_id: str, new_role: UserRole) -> bool:
        """Update a user's role"""
        user = self.users.get(user_id)
        if user:
            user.role = new_role
            return True
        return False
    
    def deactivate_user(self, user_id: str) -> bool:
        """Deactivate a user"""
        user = self.users.get(user_id)
        if user:
            user.is_active = False
            return True
        return False
    
    def activate_user(self, user_id: str) -> bool:
        """Activate a user"""
        user = self.users.get(user_id)
        if user:
            user.is_active = True
            return True
        return False
    
    def check_permission(self, user_id: str, permission: Permission) -> bool:
        """
        Check if a user has a specific permission.
        
        Returns False if user doesn't exist or is inactive.
        """
        user = self.users.get(user_id)
        if not user or not user.is_active:
            return False
        return user.has_permission(permission)
    
    def list_users_by_role(self, role: UserRole) -> List[User]:
        """Get all users with a specific role"""
        return [u for u in self.users.values() if u.role == role]
    
    def get_user_statistics(self) -> Dict:
        """Get statistics about users in the system"""
        stats = {
            'total_users': len(self.users),
            'active_users': len([u for u in self.users.values() if u.is_active]),
            'by_role': {}
        }
        
        for role in UserRole:
            count = len([u for u in self.users.values() if u.role == role])
            stats['by_role'][role.value] = count
        
        return stats
