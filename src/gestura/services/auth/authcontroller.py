import hashlib 

class AuthController:
    def __init__(self, db_manager):
        self.db = db_manager
        
    def _hash_password(self, password):
        return hashlib.sha256(password.encode('utf-8')).hexdigest()

    def authenticate(self, username, password):
        user_record = self.db.auth_user(username)
        
        if user_record is None:
            return False 
        
        stored_hash = user_record[0]
        input_hash = self._hash_password(password)
        
        return stored_hash == input_hash
        
    def register_user(self, username, password):
        if not username or not password: 
            return False 
        
        hashed_password = self._hash_password(password)
        
        return self.db.create_user(username, hashed_password)