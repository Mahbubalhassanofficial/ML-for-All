import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
import io
import base64
import warnings
warnings.filterwarnings('ignore')

# Try to import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Set page config
st.set_page_config(page_title="AutoML Research Platform", layout="wide", page_icon="🤖")

# Configure matplotlib for publication-quality figures
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.dpi'] = 600

def save_figure_high_quality(fig, filename):
    """Save figure in publication quality"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=600, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    return buf

def detect_problem_type(df, target_column):
    """Automatically detect if it's classification, regression, or clustering"""
    if target_column is None or target_column == "":
        return "clustering"
    
    target = df[target_column]
    unique_ratio = len(target.unique()) / len(target)
    
    # If target has few unique values relative to dataset size, it's classification
    if unique_ratio < 0.05 or len(target.unique()) <= 20:
        return "classification"
    else:
        return "regression"

def recommend_models(problem_type, n_samples, n_features):
    """Recommend models based on problem type and data characteristics"""
    recommendations = {
        "classification": {
            "small": ["Logistic Regression", "Decision Tree", "Naive Bayes"],
            "medium": ["Random Forest", "Gradient Boosting", "SVM"],
            "large": ["Random Forest", "Gradient Boosting", "Logistic Regression"]
        },
        "regression": {
            "small": ["Linear Regression", "Ridge", "Lasso"],
            "medium": ["Random Forest", "Gradient Boosting", "SVR"],
            "large": ["Random Forest", "Gradient Boosting", "Ridge"]
        },
        "clustering": {
            "small": ["K-Means", "Hierarchical"],
            "medium": ["K-Means", "DBSCAN"],
            "large": ["K-Means", "DBSCAN"]
        }
    }
    
    if n_samples < 1000:
        size = "small"
    elif n_samples < 10000:
        size = "medium"
    else:
        size = "large"
    
    return recommendations[problem_type][size]

def get_model_parameters(model_name):
    """Get parameter grids for hyperparameter tuning"""
    params = {
        "Random Forest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        },
        "Gradient Boosting": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.3],
            "max_depth": [3, 5, 7],
            "subsample": [0.8, 1.0]
        },
        "Logistic Regression": {
            "C": [0.001, 0.01, 0.1, 1, 10, 100],
            "penalty": ['l1', 'l2'],
            "solver": ['liblinear', 'saga']
        },
        "SVM": {
            "C": [0.1, 1, 10, 100],
            "kernel": ['rbf', 'linear', 'poly'],
            "gamma": ['scale', 'auto', 0.001, 0.01]
        },
        "Ridge": {
            "alpha": [0.001, 0.01, 0.1, 1, 10, 100]
        },
        "Lasso": {
            "alpha": [0.001, 0.01, 0.1, 1, 10, 100]
        },
        "Decision Tree": {
            "max_depth": [None, 5, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        },
        "K-Nearest Neighbors": {
            "n_neighbors": [3, 5, 7, 9, 11],
            "weights": ['uniform', 'distance'],
            "metric": ['euclidean', 'manhattan']
        }
    }
    return params.get(model_name, {})

def create_model(model_name, problem_type, params=None):
    """Create model instance based on name and type"""
    models = {
        "classification": {
            "Random Forest": RandomForestClassifier(random_state=42, **params) if params else RandomForestClassifier(random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42, **params) if params else GradientBoostingClassifier(random_state=42),
            "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000, **params) if params else LogisticRegression(random_state=42, max_iter=1000),
            "SVM": SVC(random_state=42, probability=True, **params) if params else SVC(random_state=42, probability=True),
            "Decision Tree": DecisionTreeClassifier(random_state=42, **params) if params else DecisionTreeClassifier(random_state=42),
            "K-Nearest Neighbors": KNeighborsClassifier(**params) if params else KNeighborsClassifier(),
            "Naive Bayes": GaussianNB()
        },
        "regression": {
            "Random Forest": RandomForestRegressor(random_state=42, **params) if params else RandomForestRegressor(random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42, **params) if params else GradientBoostingRegressor(random_state=42),
            "Linear Regression": LinearRegression(),
            "Ridge": Ridge(random_state=42, **params) if params else Ridge(random_state=42),
            "Lasso": Lasso(random_state=42, **params) if params else Lasso(random_state=42),
            "SVR": SVR(**params) if params else SVR(),
            "Decision Tree": DecisionTreeRegressor(random_state=42, **params) if params else DecisionTreeRegressor(random_state=42),
            "K-Nearest Neighbors": KNeighborsRegressor(**params) if params else KNeighborsRegressor()
        },
        "clustering": {
            "K-Means": KMeans(random_state=42),
            "DBSCAN": DBSCAN(),
            "Hierarchical": AgglomerativeClustering()
        }
    }
    return models[problem_type][model_name]

def plot_confusion_matrix(cm, class_names):
    """Create publication-quality confusion matrix"""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    return fig

def plot_feature_importance(model, feature_names, top_n=20):
    """Create publication-quality feature importance plot"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(indices)), importances[indices])
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Feature Importance')
        ax.invert_yaxis()
        plt.tight_layout()
        return fig
    return None

def plot_regression_results(y_true, y_pred):
    """Create publication-quality regression plot"""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.5, edgecolors='k', linewidth=0.5)
    
    # Add perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.legend()
    plt.tight_layout()
    return fig

def plot_residuals(y_true, y_pred):
    """Create publication-quality residual plot"""
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_pred, residuals, alpha=0.5, edgecolors='k', linewidth=0.5)
    ax.axhline(y=0, color='r', linestyle='--', lw=2)
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Residuals')
    plt.tight_layout()
    return fig

def plot_clustering_results(X, labels, n_components=2):
    """Create publication-quality clustering visualization"""
    from sklearn.decomposition import PCA
    
    if X.shape[1] > 2:
        pca = PCA(n_components=n_components)
        X_reduced = pca.fit_transform(X)
    else:
        X_reduced = X
    
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], 
                        c=labels, cmap='viridis', 
                        alpha=0.6, edgecolors='k', linewidth=0.5)
    ax.set_xlabel('First Principal Component' if X.shape[1] > 2 else 'Feature 1')
    ax.set_ylabel('Second Principal Component' if X.shape[1] > 2 else 'Feature 2')
    plt.colorbar(scatter, ax=ax, label='Cluster')
    plt.tight_layout()
    return fig

def plot_roc_curve(y_test, y_pred_proba, class_names):
    """Create publication-quality ROC curve"""
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    
    n_classes = len(class_names)
    
    if n_classes == 2:
        # Binary classification
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend()
        plt.tight_layout()
        return fig
    else:
        # Multiclass
        y_test_bin = label_binarize(y_test, classes=range(n_classes))
        
        fig, ax = plt.subplots(figsize=(8, 6))
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, label=f'{class_names[i]} (AUC = {roc_auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend()
        plt.tight_layout()
        return fig

def plot_learning_curve(estimator, X, y, cv=5):
    """Create publication-quality learning curve"""
    from sklearn.model_selection import learning_curve
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy' if hasattr(estimator, 'predict_proba') else 'r2'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(train_sizes, train_mean, 'o-', lw=2, label='Training score')
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
    ax.plot(train_sizes, test_mean, 'o-', lw=2, label='Cross-validation score')
    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.2)
    ax.set_xlabel('Training Examples')
    ax.set_ylabel('Score')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def results_to_latex(results_df, caption="Model Performance Metrics"):
    """Convert results to LaTeX table format"""
    latex_str = results_df.to_latex(
        index=False,
        float_format="%.4f",
        caption=caption,
        label="tab:results",
        position="htbp"
    )
    return latex_str

def cv_results_to_latex(cv_results, model_name):
    """Convert cross-validation results to LaTeX table"""
    cv_df = pd.DataFrame({
        'Fold': [f'Fold {i+1}' for i in range(len(cv_results))] + ['Mean', 'Std'],
        'Score': list(cv_results) + [np.mean(cv_results), np.std(cv_results)]
    })
    
    latex_str = cv_df.to_latex(
        index=False,
        float_format="%.4f",
        caption=f"Cross-Validation Results for {model_name}",
        label="tab:cv_results",
        position="htbp"
    )
    return latex_str

# Main App
st.title("🤖 AutoML Research Platform")
st.markdown("### Professional Machine Learning Pipeline for Research")

# Sidebar - Workflow Guide
with st.sidebar:
    st.header("📚 Workflow Guide")
    
    workflow_step = st.selectbox(
        "Select Workflow Step",
        ["Overview", "Data Upload", "Problem Detection", "Model Selection", 
         "Hyperparameter Tuning", "Cross-Validation", "Training & Evaluation", 
         "Interpretability", "Export Results"]
    )
    
    if workflow_step == "Overview":
        st.info("""
        **AutoML Workflow:**
        1. Upload your dataset (CSV/Excel)
        2. System detects problem type
        3. Select target variable
        4. Choose recommended models
        5. Optional: Hyperparameter tuning
        6. Optional: Cross-validation
        7. Train and evaluate
        8. SHAP interpretability
        9. Export LaTeX tables & figures
        """)
    elif workflow_step == "Data Upload":
        st.info("""
        **Data Guidelines:**
        - Supported: CSV, Excel
        - Clean data preferred
        - Handle missing values
        - Numeric features work best
        """)
    elif workflow_step == "Problem Detection":
        st.info("""
        **Auto-Detection:**
        - Classification: < 20 unique target values
        - Regression: Continuous target values
        - Clustering: No target variable
        """)
    elif workflow_step == "Model Selection":
        st.info("""
        **Model Recommendations:**
        - Small data (< 1k): Simple models
        - Medium data (1k-10k): Ensemble methods
        - Large data (> 10k): Scalable algorithms
        """)
    elif workflow_step == "Hyperparameter Tuning":
        st.info("""
        **Grid Search CV:**
        - Exhaustive search over parameters
        - Cross-validation for each combination
        - Finds optimal parameters automatically
        - May take longer for large datasets
        """)
    elif workflow_step == "Cross-Validation":
        st.info("""
        **K-Fold Cross-Validation:**
        - More robust performance estimate
        - Reduces overfitting risk
        - Recommended: 5 or 10 folds
        - Shows model stability
        """)
    elif workflow_step == "Interpretability":
        st.info("""
        **SHAP Analysis:**
        - Explains individual predictions
        - Shows feature contributions
        - Identifies important features
        - Helps understand model decisions
        """)
    elif workflow_step == "Export Results":
        st.info("""
        **Export Options:**
        - LaTeX tables for papers
        - High-res figures (600 DPI PNG)
        - CSV results for further analysis
        - Publication-ready formatting
        """)

# Main Content Area
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Data Upload", "🔬 Model Training", "📈 Results", "🔍 Interpretability", "📤 Export"])

with tab1:
    st.header("Data Upload & Exploration")
    
    uploaded_file = st.file_uploader("Upload your dataset", type=['csv', 'xlsx', 'xls'])
    
    if uploaded_file is not None:
        # Load data
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Store in session state
            st.session_state['df'] = df
            
            # Data preview
            st.subheader("Data Preview")
            st.dataframe(df.head(10))
            
            # Basic statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Rows", df.shape[0])
                st.metric("Columns", df.shape[1])
                
            with col2:
                st.metric("Missing Values", df.isnull().sum().sum())
                st.metric("Duplicates", df.duplicated().sum())
                
            with col3:
                st.metric("Numeric Columns", len(df.select_dtypes(include=[np.number]).columns))
                st.metric("Categorical Columns", len(df.select_dtypes(include=['object']).columns))
            
            # Statistical summary
            st.subheader("Statistical Summary")
            st.dataframe(df.describe())
            
            # Missing values visualization
            if df.isnull().sum().sum() > 0:
                st.subheader("Missing Values Distribution")
                missing_data = df.isnull().sum()
                missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(range(len(missing_data)), missing_data.values)
                ax.set_yticks(range(len(missing_data)))
                ax.set_yticklabels(missing_data.index)
                ax.set_xlabel('Number of Missing Values')
                ax.invert_yaxis()
                plt.tight_layout()
                st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Error loading file: {e}")

with tab2:
    st.header("Model Training & Configuration")
    
    if 'df' in st.session_state:
        df = st.session_state['df']
        
        # Problem type detection
        st.subheader("1. Define Problem")
        
        col1, col2 = st.columns(2)
        
        with col1:
            target_column = st.selectbox(
                "Select Target Variable (leave empty for clustering)",
                [""] + list(df.columns)
            )
        
        if target_column and target_column != "":
            problem_type = detect_problem_type(df, target_column)
            st.session_state['problem_type'] = problem_type
            st.session_state['target_column'] = target_column
            
            with col2:
                st.info(f"🎯 Detected Problem: **{problem_type.upper()}**")
            
            # Feature selection
            st.subheader("2. Feature Selection")
            available_features = [col for col in df.columns if col != target_column]
            selected_features = st.multiselect(
                "Select Features (leave empty to use all)",
                available_features,
                default=available_features
            )
            
            if not selected_features:
                selected_features = available_features
            
            st.session_state['selected_features'] = selected_features
            
            # Model recommendation
            st.subheader("3. Model Selection")
            recommended_models = recommend_models(
                problem_type, 
                len(df), 
                len(selected_features)
            )
            
            st.success(f"📌 Recommended Models: {', '.join(recommended_models)}")
            
            # Model selection
            if problem_type == "classification":
                all_models = ["Random Forest", "Gradient Boosting", "Logistic Regression", 
                             "SVM", "Decision Tree", "K-Nearest Neighbors", "Naive Bayes"]
            elif problem_type == "regression":
                all_models = ["Random Forest", "Gradient Boosting", "Linear Regression",
                             "Ridge", "Lasso", "SVR", "Decision Tree", "K-Nearest Neighbors"]
            
            selected_model = st.selectbox("Choose Model", all_models)
            
            # Advanced options
            st.subheader("4. Advanced Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                use_hyperparameter_tuning = st.checkbox("Enable Hyperparameter Tuning (Grid Search)")
                use_cross_validation = st.checkbox("Enable Cross-Validation", value=True)
                
            with col2:
                if use_cross_validation:
                    cv_folds = st.slider("Number of CV Folds", 3, 10, 5)
                test_size = st.slider("Test Set Size (%)", 10, 40, 20) / 100
            
            # Train button
            if st.button("🚀 Train Model", type="primary"):
                with st.spinner("Training model... This may take a few minutes."):
                    try:
                        # Prepare data
                        X = df[selected_features].copy()
                        y = df[target_column].copy()
                        
                        # Handle categorical variables
                        label_encoders = {}
                        for col in X.select_dtypes(include=['object']).columns:
                            le = LabelEncoder()
                            X[col] = le.fit_transform(X[col].astype(str))
                            label_encoders[col] = le
                        
                        # Handle categorical target for classification
                        le_target = None
                        if problem_type == "classification" and y.dtype == 'object':
                            le_target = LabelEncoder()
                            y = le_target.fit_transform(y)
                            st.session_state['label_encoder'] = le_target
                        
                        # Split data
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size, random_state=42
                        )
                        
                        # Scale features
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        
                        # Create base model
                        base_model = create_model(selected_model, problem_type)
                        
                        # Hyperparameter tuning
                        if use_hyperparameter_tuning and selected_model in get_model_parameters(selected_model):
                            st.info("🔍 Performing hyperparameter tuning...")
                            param_grid = get_model_parameters(selected_model)
                            
                            grid_search = GridSearchCV(
                                base_model,
                                param_grid,
                                cv=cv_folds if use_cross_validation else 5,
                                scoring='accuracy' if problem_type == 'classification' else 'r2',
                                n_jobs=-1,
                                verbose=0
                            )
                            
                            grid_search.fit(X_train_scaled, y_train)
                            model = grid_search.best_estimator_
                            
                            st.success(f"✅ Best parameters found: {grid_search.best_params_}")
                            st.session_state['best_params'] = grid_search.best_params_
                            st.session_state['grid_search_results'] = pd.DataFrame(grid_search.cv_results_)
                        else:
                            model = base_model
                            model.fit(X_train_scaled, y_train)
                        
                        # Cross-validation
                        if use_cross_validation:
                            st.info("📊 Performing cross-validation...")
                            cv_scores = cross_val_score(
                                model, X_train_scaled, y_train, 
                                cv=cv_folds,
                                scoring='accuracy' if problem_type == 'classification' else 'r2'
                            )
                            st.session_state['cv_scores'] = cv_scores
                            st.success(f"✅ CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
                        
                        # Predictions
                        y_pred = model.predict(X_test_scaled)
                        
                        # Prediction probabilities for classification
                        if problem_type == "classification" and hasattr(model, 'predict_proba'):
                            y_pred_proba = model.predict_proba(X_test_scaled)
                            st.session_state['y_pred_proba'] = y_pred_proba
                        
                        # Store results
                        st.session_state['model'] = model
                        st.session_state['X_train'] = X_train_scaled
                        st.session_state['X_test'] = X_test_scaled
                        st.session_state['y_train'] = y_train
                        st.session_state['y_test'] = y_test
                        st.session_state['y_pred'] = y_pred
                        st.session_state['scaler'] = scaler
                        st.session_state['feature_names'] = selected_features
                        st.session_state['selected_model'] = selected_model
                        st.session_state['label_encoders'] = label_encoders
                        
                        st.success("✅ Model trained successfully!")
                        st.balloons()
                        
                    except Exception as e:
                        st.error(f"Training error: {e}")
                        import traceback
                        st.error(traceback.format_exc())
        
        else:
            # Clustering workflow
            st.session_state['problem_type'] = "clustering"
            st.info("🎯 **Clustering Mode** - No target variable needed")
            
            st.subheader("2. Feature Selection")
            selected_features = st.multiselect(
                "Select Features for Clustering",
                list(df.columns),
                default=list(df.columns)[:min(10, len(df.columns))]
            )
            
            if selected_features:
                st.session_state['selected_features'] = selected_features
                
                st.subheader("3. Clustering Configuration")
                cluster_model = st.selectbox("Choose Clustering Algorithm", 
                                            ["K-Means", "DBSCAN", "Hierarchical"])
                
                if cluster_model == "K-Means":
                    n_clusters = st.slider("Number of Clusters", 2, 10, 3)
                elif cluster_model == "Hierarchical":
                    n_clusters = st.slider("Number of Clusters", 2, 10, 3)
                
                if st.button("🚀 Perform Clustering", type="primary"):
                    with st.spinner("Clustering..."):
                        try:
                            X = df[selected_features].copy()
                            
                            # Handle categorical variables
                            for col in X.select_dtypes(include=['object']).columns:
                                le = LabelEncoder()
                                X[col] = le.fit_transform(X[col].astype(str))
                            
                            # Scale features
                            scaler = StandardScaler()
                            X_scaled = scaler.fit_transform(X)
                            
                            # Clustering
                            if cluster_model == "K-Means":
                                model = KMeans(n_clusters=n_clusters, random_state=42)
                            elif cluster_model == "DBSCAN":
                                model = DBSCAN(eps=0.5, min_samples=5)
                            else:
                                model = AgglomerativeClustering(n_clusters=n_clusters)
                            
                            labels = model.fit_predict(X_scaled)
                            
                            # Store results
                            st.session_state['model'] = model
                            st.session_state['X_scaled'] = X_scaled
                            st.session_state['labels'] = labels
                            st.session_state['scaler'] = scaler
                            st.session_state['feature_names'] = selected_features
                            st.session_state['selected_model'] = cluster_model
                            
                            st.success("✅ Clustering completed!")
                            st.balloons()
                            
                        except Exception as e:
                            st.error(f"Clustering error: {e}")
    
    else:
        st.warning("⚠️ Please upload a dataset first")

with tab3:
    st.header("Results & Visualizations")
    
    if 'model' in st.session_state:
        problem_type = st.session_state['problem_type']
        
        if problem_type == "classification":
            y_test = st.session_state['y_test']
            y_pred = st.session_state['y_pred']
            
            # Metrics
            st.subheader("📊 Performance Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
            col2.metric("Precision", f"{precision_score(y_test, y_pred, average='weighted'):.4f}")
            col3.metric("Recall", f"{recall_score(y_test, y_pred, average='weighted'):.4f}")
            col4.metric("F1-Score", f"{f1_score(y_test, y_pred, average='weighted'):.4f}")
            
            # Cross-validation results
            if 'cv_scores' in st.session_state:
                st.subheader("🔄 Cross-Validation Results")
                cv_scores = st.session_state['cv_scores']
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Mean CV Score", f"{cv_scores.mean():.4f}")
                col2.metric("Std CV Score", f"{cv_scores.std():.4f}")
                col3.metric("Min-Max CV", f"{cv_scores.min():.4f} - {cv_scores.max():.4f}")
                
                # CV scores table
                cv_df = pd.DataFrame({
                    'Fold': [f'Fold {i+1}' for i in range(len(cv_scores))],
                    'Score': cv_scores
                })
                st.dataframe(cv_df)
            
            # Results table
            st.subheader("📋 Detailed Results")
            results_df = pd.DataFrame({
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                'Score': [
                    accuracy_score(y_test, y_pred),
                    precision_score(y_test, y_pred, average='weighted'),
                    recall_score(y_test, y_pred, average='weighted'),
                    f1_score(y_test, y_pred, average='weighted')
                ]
            })
            st.dataframe(results_df)
            st.session_state['results_df'] = results_df
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Confusion Matrix
                st.subheader("🎯 Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                
                if 'label_encoder' in st.session_state:
                    class_names = st.session_state['label_encoder'].classes_
                else:
                    class_names = [f"Class {i}" for i in range(len(np.unique(y_test)))]
                
                fig_cm = plot_confusion_matrix(cm, class_names)
                st.pyplot(fig_cm)
                st.session_state['fig_cm'] = fig_cm
            
            with col2:
                # ROC Curve
                if 'y_pred_proba' in st.session_state:
                    st.subheader("📈 ROC Curve")
                    fig_roc = plot_roc_curve(y_test, st.session_state['y_pred_proba'], class_names)
                    st.pyplot(fig_roc)
                    st.session_state['fig_roc'] = fig_roc
            
            # Feature Importance
            if hasattr(st.session_state['model'], 'feature_importances_'):
                st.subheader("📊 Feature Importance")
                fig_fi = plot_feature_importance(
                    st.session_state['model'], 
                    st.session_state['feature_names']
                )
                if fig_fi:
                    st.pyplot(fig_fi)
                    st.session_state['fig_fi'] = fig_fi
            
            # Learning Curve
            st.subheader("📚 Learning Curve")
            with st.spinner("Generating learning curve..."):
                X_train = st.session_state['X_train']
                y_train = st.session_state['y_train']
                fig_lc = plot_learning_curve(st.session_state['model'], X_train, y_train)
                st.pyplot(fig_lc)
                st.session_state['fig_lc'] = fig_lc
        
        elif problem_type == "regression":
            y_test = st.session_state['y_test']
            y_pred = st.session_state['y_pred']
            
            # Metrics
            st.subheader("📊 Performance Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("R² Score", f"{r2_score(y_test, y_pred):.4f}")
            col2.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
            col3.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.4f}")
            col4.metric("MSE", f"{mean_squared_error(y_test, y_pred):.4f}")
            
            # Cross-validation results
            if 'cv_scores' in st.session_state:
                st.subheader("🔄 Cross-Validation Results")
                cv_scores = st.session_state['cv_scores']
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Mean CV Score", f"{cv_scores.mean():.4f}")
                col2.metric("Std CV Score", f"{cv_scores.std():.4f}")
                col3.metric("Min-Max CV", f"{cv_scores.min():.4f} - {cv_scores.max():.4f}")
            
            # Results table
            st.subheader("📋 Detailed Results")
            results_df = pd.DataFrame({
                'Metric': ['R² Score', 'RMSE', 'MAE', 'MSE'],
                'Score': [
                    r2_score(y_test, y_pred),
                    np.sqrt(mean_squared_error(y_test, y_pred)),
                    mean_absolute_error(y_test, y_pred),
                    mean_squared_error(y_test, y_pred)
                ]
            })
            st.dataframe(results_df)
            st.session_state['results_df'] = results_df
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Prediction plot
                st.subheader("📈 Actual vs Predicted")
                fig_reg = plot_regression_results(y_test, y_pred)
                st.pyplot(fig_reg)
                st.session_state['fig_reg'] = fig_reg
            
            with col2:
                # Residual plot
                st.subheader("📉 Residual Plot")
                fig_res = plot_residuals(y_test, y_pred)
                st.pyplot(fig_res)
                st.session_state['fig_res'] = fig_res
            
            # Feature Importance
            if hasattr(st.session_state['model'], 'feature_importances_'):
                st.subheader("📊 Feature Importance")
                fig_fi = plot_feature_importance(
                    st.session_state['model'], 
                    st.session_state['feature_names']
                )
                if fig_fi:
                    st.pyplot(fig_fi)
                    st.session_state['fig_fi'] = fig_fi
            
            # Learning Curve
            st.subheader("📚 Learning Curve")
            with st.spinner("Generating learning curve..."):
                X_train = st.session_state['X_train']
                y_train = st.session_state['y_train']
                fig_lc = plot_learning_curve(st.session_state['model'], X_train, y_train)
                st.pyplot(fig_lc)
                st.session_state['fig_lc'] = fig_lc
        
        elif problem_type == "clustering":
            labels = st.session_state['labels']
            X_scaled = st.session_state['X_scaled']
            
            # Metrics
            st.subheader("📊 Clustering Metrics")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Number of Clusters", len(np.unique(labels)))
            col2.metric("Silhouette Score", f"{silhouette_score(X_scaled, labels):.4f}")
            col3.metric("Davies-Bouldin Index", f"{davies_bouldin_score(X_scaled, labels):.4f}")
            
            # Cluster sizes
            st.subheader("📋 Cluster Distribution")
            cluster_counts = pd.Series(labels).value_counts().sort_index()
            cluster_df = pd.DataFrame({
                'Cluster': cluster_counts.index,
                'Size': cluster_counts.values,
                'Percentage': (cluster_counts.values / len(labels) * 100).round(2)
            })
            st.dataframe(cluster_df)
            st.session_state['cluster_df'] = cluster_df
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Bar chart
                fig_bar, ax = plt.subplots(figsize=(8, 6))
                ax.bar(cluster_df['Cluster'], cluster_df['Size'])
                ax.set_xlabel('Cluster')
                ax.set_ylabel('Number of Samples')
                plt.tight_layout()
                st.pyplot(fig_bar)
                st.session_state['fig_bar'] = fig_bar
            
            with col2:
                # Visualization
                st.subheader("🎨 Cluster Visualization")
                fig_cluster = plot_clustering_results(X_scaled, labels)
                st.pyplot(fig_cluster)
                st.session_state['fig_cluster'] = fig_cluster
    
    else:
        st.warning("⚠️ Please train a model first")

with tab4:
    st.header("Model Interpretability (SHAP)")
    
    if 'model' in st.session_state and st.session_state.get('problem_type') != 'clustering':
        
        if SHAP_AVAILABLE:
            st.info("🔍 **SHAP Analysis** - Understanding model predictions")
            
            st.markdown("""
            SHAP (SHapley Additive exPlanations) values show the contribution of each feature 
            to the model's predictions. This helps understand which features are most important 
            for individual predictions and overall model behavior.
            """)
            
            shap_type = st.radio("Select SHAP Analysis Type", 
                               ["Summary Plot", "Feature Importance", "Individual Prediction"])
            
            if st.button("Generate SHAP Analysis"):
                with st.spinner("Calculating SHAP values... This may take a few minutes."):
                    try:
                        X_test = st.session_state['X_test']
                        model = st.session_state['model']
                        feature_names = st.session_state['feature_names']
                        
                        # Create SHAP explainer
                        if st.session_state['selected_model'] in ['Random Forest', 'Gradient Boosting', 'Decision Tree']:
                            explainer = shap.TreeExplainer(model)
                            shap_values = explainer.shap_values(X_test)
                        else:
                            # Use KernelExplainer for other models (slower)
                            explainer = shap.KernelExplainer(model.predict, X_test[:100])
                            shap_values = explainer.shap_values(X_test[:100])
                        
                        if shap_type == "Summary Plot":
                            st.subheader("📊 SHAP Summary Plot")
                            fig_shap, ax = plt.subplots(figsize=(10, 8))
                            
                            if isinstance(shap_values, list):
                                # Multi-class classification
                                shap.summary_plot(shap_values[1], X_test, 
                                                feature_names=feature_names, 
                                                show=False)
                            else:
                                shap.summary_plot(shap_values, X_test, 
                                                feature_names=feature_names, 
                                                show=False)
                            
                            st.pyplot(fig_shap)
                            st.session_state['fig_shap_summary'] = fig_shap
                            
                        elif shap_type == "Feature Importance":
                            st.subheader("📈 SHAP Feature Importance")
                            fig_shap, ax = plt.subplots(figsize=(10, 8))
                            
                            if isinstance(shap_values, list):
                                shap.summary_plot(shap_values[1], X_test, 
                                                feature_names=feature_names,
                                                plot_type="bar", show=False)
                            else:
                                shap.summary_plot(shap_values, X_test, 
                                                feature_names=feature_names,
                                                plot_type="bar", show=False)
                            
                            st.pyplot(fig_shap)
                            st.session_state['fig_shap_importance'] = fig_shap
                            
                        else:  # Individual Prediction
                            st.subheader("🔍 Individual Prediction Explanation")
                            sample_idx = st.number_input("Select sample index", 
                                                        min_value=0, 
                                                        max_value=len(X_test)-1, 
                                                        value=0)
                            
                            fig_shap, ax = plt.subplots(figsize=(10, 6))
                            
                            if isinstance(shap_values, list):
                                shap.waterfall_plot(
                                    shap.Explanation(values=shap_values[1][sample_idx],
                                                   base_values=explainer.expected_value[1],
                                                   data=X_test[sample_idx],
                                                   feature_names=feature_names),
                                    show=False
                                )
                            else:
                                shap.waterfall_plot(
                                    shap.Explanation(values=shap_values[sample_idx],
                                                   base_values=explainer.expected_value,
                                                   data=X_test[sample_idx],
                                                   feature_names=feature_names),
                                    show=False
                                )
                            
                            st.pyplot(fig_shap)
                            st.session_state['fig_shap_individual'] = fig_shap
                        
                        st.success("✅ SHAP analysis completed!")
                        
                    except Exception as e:
                        st.error(f"SHAP analysis error: {e}")
                        st.info("Try using a tree-based model (Random Forest, Gradient Boosting) for faster SHAP computation.")
        
        else:
            st.warning("⚠️ SHAP library not installed. Install it with: `pip install shap`")
            st.info("For now, feature importance plots are available in the Results tab.")
    
    else:
        st.warning("⚠️ Interpretability analysis is available after model training (not for clustering)")

with tab5:
    st.header("Export Results")
    
    if 'model' in st.session_state:
        st.success("✅ Results are ready to export!")
        
        # Export options
        st.subheader("📥 Export Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Tables")
            
            # Export results table as CSV
            if 'results_df' in st.session_state:
                csv = st.session_state['results_df'].to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name="model_results.csv",
                    mime="text/csv"
                )
            
            # Export results as LaTeX
            if 'results_df' in st.session_state:
                latex_results = results_to_latex(st.session_state['results_df'])
                st.download_button(
                    label="Download Results as LaTeX",
                    data=latex_results,
                    file_name="model_results.tex",
                    mime="text/plain"
                )
            
            # Export CV results as LaTeX
            if 'cv_scores' in st.session_state:
                latex_cv = cv_results_to_latex(
                    st.session_state['cv_scores'],
                    st.session_state['selected_model']
                )
                st.download_button(
                    label="Download CV Results as LaTeX",
                    data=latex_cv,
                    file_name="cv_results.tex",
                    mime="text/plain"
                )
            
            # Export cluster distribution
            if 'cluster_df' in st.session_state:
                cluster_csv = st.session_state['cluster_df'].to_csv(index=False)
                st.download_button(
                    label="Download Cluster Distribution as CSV",
                    data=cluster_csv,
                    file_name="cluster_distribution.csv",
                    mime="text/csv"
                )
                
                latex_cluster = results_to_latex(
                    st.session_state['cluster_df'],
                    "Cluster Distribution"
                )
                st.download_button(
                    label="Download Cluster Distribution as LaTeX",
                    data=latex_cluster,
                    file_name="cluster_distribution.tex",
                    mime="text/plain"
                )
        
        with col2:
            st.markdown("### 🖼️ Figures (600 DPI)")
            
            # Export all figures
            figure_names = {
                'fig_cm': 'confusion_matrix',
                'fig_roc': 'roc_curve',
                'fig_fi': 'feature_importance',
                'fig_lc': 'learning_curve',
                'fig_reg': 'regression_plot',
                'fig_res': 'residual_plot',
                'fig_cluster': 'clustering',
                'fig_bar': 'cluster_distribution',
                'fig_shap_summary': 'shap_summary',
                'fig_shap_importance': 'shap_importance',
                'fig_shap_individual': 'shap_individual'
            }
            
            for fig_key, fig_name in figure_names.items():
                if fig_key in st.session_state:
                    buf = save_figure_high_quality(
                        st.session_state[fig_key], 
                        f"{fig_name}.png"
                    )
                    st.download_button(
                        label=f"Download {fig_name.replace('_', ' ').title()}",
                        data=buf,
                        file_name=f"{fig_name}_600dpi.png",
                        mime="image/png"
                    )
        
        # LaTeX template
        st.subheader("📄 LaTeX Integration Guide")
        
        with st.expander("View LaTeX Template"):
            latex_template = r"""
\documentclass{article}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{float}

\begin{document}

\section{Results}

% Include your results table
\input{model_results.tex}

% Include figure
\begin{figure}[H]
    \centering
    \includegraphics[width=0.8\textwidth]{confusion_matrix_600dpi.png}
    \caption{Confusion matrix showing model performance.}
    \label{fig:confusion_matrix}
\end{figure}

% Include cross-validation results
\input{cv_results.tex}

\end{document}
            """
            st.code(latex_template, language='latex')
        
        # Model summary
        st.subheader("📝 Model Summary")
        
        summary_text = f"""
**Model Information:**
- Model Type: {st.session_state['selected_model']}
- Problem Type: {st.session_state['problem_type'].title()}
- Number of Features: {len(st.session_state['feature_names'])}
- Features: {', '.join(st.session_state['feature_names'])}
"""
        
        if 'best_params' in st.session_state:
            summary_text += f"\n**Best Parameters:**\n"
            for param, value in st.session_state['best_params'].items():
                summary_text += f"- {param}: {value}\n"
        
        if 'cv_scores' in st.session_state:
            cv = st.session_state['cv_scores']
            summary_text += f"\n**Cross-Validation:**\n"
            summary_text += f"- Mean Score: {cv.mean():.4f}\n"
            summary_text += f"- Std Score: {cv.std():.4f}\n"
        
        st.markdown(summary_text)
        
        # Download model summary
        st.download_button(
            label="Download Model Summary",
            data=summary_text,
            file_name="model_summary.txt",
            mime="text/plain"
        )
    
    else:
        st.warning("⚠️ Please train a model first to export results")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>AutoML Research Platform</strong> | Built for researchers and data scientists</p>
    <p>Publication-ready outputs: 600 DPI PNG figures | LaTeX tables | Times New Roman fonts</p>
</div>
""", unsafe_allow_html=True)
