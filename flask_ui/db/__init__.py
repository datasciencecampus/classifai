"""
DATABASE.

2 Purposes for this Database:
1) To save the state of the applciation for a given user, serving as a backup in fault cases.
2) To save valuable data for internal evaluation purposes

How user details are acquired:
user_email = request.headers.get("X-Goog-Authenticated-User-Email")
user_id = request.headers.get("X-Goog-Authenticated-User-ID")
"""

from flask_ui.db.models import db

sqlite_tmp_config = "sqlite:////tmp/classifai.db"
sqlite_app_config = "sqlite:///classifai.db"
db = db
