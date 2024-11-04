-- Schema for the 'users' table to store user details
CREATE TABLE users (
    user_id BIGINT PRIMARY KEY,
    username VARCHAR(255) NOT NULL UNIQUE,
    email VARCHAR(255) NOT NULL UNIQUE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'active',
    last_login TIMESTAMP NULL,
    CONSTRAINT chk_user_status CHECK (status IN ('active', 'inactive', 'banned'))
);

-- Schema for the 'items' table to store information about products or items
CREATE TABLE items (
    item_id BIGINT PRIMARY KEY,
    item_name VARCHAR(255) NOT NULL,
    item_description TEXT,
    category_id BIGINT NOT NULL,
    price DECIMAL(10, 2) NOT NULL,
    stock_quantity INT DEFAULT 0,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'available',
    CONSTRAINT chk_item_status CHECK (status IN ('available', 'unavailable', 'discontinued'))
);

-- Schema for the 'categories' table to store hierarchical item categories
CREATE TABLE categories (
    category_id BIGINT PRIMARY KEY,
    category_name VARCHAR(255) NOT NULL UNIQUE,
    parent_category_id BIGINT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NULL,
    FOREIGN KEY (parent_category_id) REFERENCES categories(category_id) ON DELETE SET NULL
);

-- Schema for the 'transactions' table to store transactional data
CREATE TABLE transactions (
    transaction_id BIGINT PRIMARY KEY,
    user_id BIGINT NOT NULL,
    item_id BIGINT NOT NULL,
    transaction_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    quantity INT NOT NULL,
    total_amount DECIMAL(10, 2) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'completed',
    CONSTRAINT chk_transaction_status CHECK (status IN ('pending', 'completed', 'failed')),
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (item_id) REFERENCES items(item_id) ON DELETE CASCADE
);

-- Schema for the 'ratings' table to store user ratings for items
CREATE TABLE ratings (
    rating_id BIGINT PRIMARY KEY,
    user_id BIGINT NOT NULL,
    item_id BIGINT NOT NULL,
    rating DECIMAL(2, 1) NOT NULL CHECK (rating >= 1.0 AND rating <= 5.0),
    review TEXT,
    rating_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (item_id) REFERENCES items(item_id) ON DELETE CASCADE
);

-- Schema for the 'user_sessions' table to track user activity on the platform
CREATE TABLE user_sessions (
    session_id BIGINT PRIMARY KEY,
    user_id BIGINT NOT NULL,
    session_start TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    session_end TIMESTAMP NULL,
    ip_address VARCHAR(50),
    device_info VARCHAR(255),
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

-- Schema for the 'inventory_logs' table to track stock changes
CREATE TABLE inventory_logs (
    log_id BIGINT PRIMARY KEY,
    item_id BIGINT NOT NULL,
    quantity_changed INT NOT NULL,
    change_reason VARCHAR(255) NOT NULL,
    log_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (item_id) REFERENCES items(item_id) ON DELETE CASCADE
);

-- Schema for the 'recommendations' table to store personalized recommendations for users
CREATE TABLE recommendations (
    recommendation_id BIGINT PRIMARY KEY,
    user_id BIGINT NOT NULL,
    recommended_items JSONB NOT NULL, -- JSON field to store recommended item IDs
    generated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

-- Schema for the 'feedback' table to store user feedback about the recommendations
CREATE TABLE feedback (
    feedback_id BIGINT PRIMARY KEY,
    user_id BIGINT NOT NULL,
    recommendation_id BIGINT NOT NULL,
    feedback_text TEXT,
    feedback_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (recommendation_id) REFERENCES recommendations(recommendation_id) ON DELETE CASCADE
);

-- Indexes for performance optimization
CREATE INDEX idx_user_email ON users (email);
CREATE INDEX idx_item_category ON items (category_id);
CREATE INDEX idx_transaction_user_date ON transactions (user_id, transaction_date);
CREATE INDEX idx_rating_user_item ON ratings (user_id, item_id);
CREATE INDEX idx_session_user ON user_sessions (user_id);
CREATE INDEX idx_inventory_item ON inventory_logs (item_id);
CREATE INDEX idx_recommendation_user ON recommendations (user_id);
CREATE INDEX idx_feedback_user_recommendation ON feedback (user_id, recommendation_id);