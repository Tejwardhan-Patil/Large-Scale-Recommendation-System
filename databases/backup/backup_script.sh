#!/bin/bash

# Configuration
DB_NAME="mydatabase"
DB_USER="myuser"
DB_HOST="localhost"
DB_PORT="5432"
BACKUP_DIR="/backup/directory"
LOG_FILE="/backup/logs/backup.log"
RETENTION_DAYS=7
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="$BACKUP_DIR/${DB_NAME}_backup_$TIMESTAMP.sql.gz"

# Logging function
log_message() {
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] - $1" >> $LOG_FILE
}

# Create backup directory if it doesn't exist
mkdir -p $BACKUP_DIR

# Backup the database
log_message "Starting backup for database: $DB_NAME"
pg_dump -h $DB_HOST -p $DB_PORT -U $DB_USER -F c $DB_NAME | gzip > $BACKUP_FILE

# Verify if backup was successful
if [ $? -eq 0 ]; then
    log_message "Backup successful: $BACKUP_FILE"
else
    log_message "Backup failed for database: $DB_NAME"
    exit 1
fi

# Retention: Delete backups older than specified retention days
log_message "Deleting backups older than $RETENTION_DAYS days"
find $BACKUP_DIR -type f -name "${DB_NAME}_backup_*.sql.gz" -mtime +$RETENTION_DAYS -exec rm {} \;

# Verify if old backups were deleted successfully
if [ $? -eq 0 ]; then
    log_message "Old backups deleted successfully"
else
    log_message "Failed to delete old backups"
fi

log_message "Backup process completed"
exit 0