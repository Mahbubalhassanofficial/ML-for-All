import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
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

def get_model_parameters(model_name, problem_type):
    """Get recommended parameters for each model"""
    params = {
        "Random Forest": {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10]
        },
        "Gradient Boosting": {
            "n_estimators": [100, 200],
            "learning_rate": [0.01, 0.1, 0.3],
            "max_depth": [3, 5, 7]
        },
        "Logistic Regression": {
            "C": [0.001, 0.01, 0.1, 1, 10],
            "penalty": ['l1', 'l2'],
            "solver": ['liblinear', 'saga']
        },
        "SVM": {
            "C": [0.1, 1, 10],
            "kernel": ['rbf', 'linear', 'poly'],
            "gamma": ['scale', 'auto']
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
            "SVM": SVC(random_state=42, **params) if params else SVC(random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "K-Nearest Neighbors": KNeighborsClassifier(),
            "Naive Bayes": GaussianNB()
        },
        "regression": {
            "Random Forest": RandomForestRegressor(random_state=42, **params) if params else RandomForestRegressor(random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42, **params) if params else GradientBoostingRegressor(random_state=42),
            "Linear Regression": LinearRegression(),
            "Ridge": Ridge(random_state=42),
            "Lasso": Lasso(random_state=42),
            "SVR": SVR(),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
            "K-Nearest Neighbors": KNeighborsRegressor()
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
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
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

# Main App
st.title("🤖 AutoML Research Platform")
st.markdown("### Professional Machine Learning Pipeline for Research")

# Sidebar - Workflow Guide
with st.sidebar:
    st.header("📚 Workflow Guide")
    
    workflow_step = st.selectbox(
        "Select Workflow Step",
        ["Overview", "Data Upload", "Problem Detection", "Model Selection", 
         "Training & Evaluation", "Interpretability", "Export Results"]
    )
    
    if workflow_step == "Overview":
        st.info("""
        **AutoML Workflow:**
        1. Upload your dataset (CSV/Excel)
        2. System detects problem type
        3. Select target variable
        4. Choose recommended models
        5. Train and evaluate
        6. Generate publication-quality results
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
        - Small data: Simple models
        - Medium data: Ensemble methods
        - Large data: Scalable algorithms
        """)

# Main Content Area
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Upload", "🔬 Model Training", "📈 Results", "🔍 Interpretability"])

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
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Dataset Info")
                st.write(f"**Rows:** {df.shape[0]}")
                st.write(f"**Columns:** {df.shape[1]}")
                st.write(f"**Missing Values:** {df.isnull().sum().sum()}")
                
            with col2:
                st.subheader("Column Types")
                st.write(df.dtypes.value_counts())
            
            # Statistical summary
            st.subheader("Statistical Summary")
            st.dataframe(df.describe())
            
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
            
            selected_model = st.selectbox("Choose Model", all_models, 
                                         index=0 if recommended_models[0] in all_models else 0)
            
            # Train button
            if st.button("🚀 Train Model", type="primary"):
                with st.spinner("Training model..."):
                    try:
                        # Prepare data
                        X = df[selected_features].copy()
                        y = df[target_column].copy()
                        
                        # Handle categorical variables
                        for col in X.select_dtypes(include=['object']).columns:
                            le = LabelEncoder()
                            X[col] = le.fit_transform(X[col].astype(str))
                        
                        # Handle categorical target for classification
                        if problem_type == "classification" and y.dtype == 'object':
                            le_target = LabelEncoder()
                            y = le_target.fit_transform(y)
                            st.session_state['label_encoder'] = le_target
                        
                        # Split data
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=42
                        )
                        
                        # Scale features
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        
                        # Create and train model
                        model = create_model(selected_model, problem_type)
                        model.fit(X_train_scaled, y_train)
                        
                        # Predictions
                        y_pred = model.predict(X_test_scaled)
                        
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
                        
                        st.success("✅ Model trained successfully!")
                        
                    except Exception as e:
                        st.error(f"Training error: {e}")
        
        else:
            # Clustering workflow
            st.session_state['problem_type'] = "clustering"
            st.info("🎯 **Clustering Mode** - No target variable needed")
            
            st.subheader("2. Feature Selection")
            selected_features = st.multiselect(
                "Select Features for Clustering",
                list(df.columns),
                default=list(df.columns)
            )
            
            if selected_features:
                st.session_state['selected_features'] = selected_features
                
                st.subheader("3. Clustering Configuration")
                cluster_model = st.selectbox("Choose Clustering Algorithm", 
                                            ["K-Means", "DBSCAN", "Hierarchical"])
                
                if cluster_model == "K-Means":
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
            
            # Confusion Matrix
            st.subheader("🎯 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            
            if 'label_encoder' in st.session_state:
                class_names = st.session_state['label_encoder'].classes_
            else:
                class_names = [f"Class {i}" for i in range(len(np.unique(y_test)))]
            
            fig_cm = plot_confusion_matrix(cm, class_names)
            st.pyplot(fig_cm)
            
            # Download button
            buf_cm = save_figure_high_quality(fig_cm, "confusion_matrix.png")
            st.download_button(
                label="📥 Download Confusion Matrix (600 DPI)",
                data=buf_cm,
                file_name="confusion_matrix_600dpi.png",
                mime="image/png"
            )
            
            # Feature Importance
            if hasattr(st.session_state['model'], 'feature_importances_'):
                st.subheader("📊 Feature Importance")
                fig_fi = plot_feature_importance(
                    st.session_state['model'], 
                    st.session_state['feature_names']
                )
                if fig_fi:
                    st.pyplot(fig_fi)
                    buf_fi = save_figure_high_quality(fig_fi, "feature_importance.png")
                    st.download_button(
                        label="📥 Download Feature Importance (600 DPI)",
                        data=buf_fi,
                        file_name="feature_importance_600dpi.png",
                        mime="image/png"
                    )
        
        elif problem_type == "regression":
            y_test = st.session_state['y_test']
            y_pred = st.session_state['y_pred']
            
            # Metrics
            st.subheader("📊 Performance Metrics")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("R² Score", f"{r2_score(y_test, y_pred):.4f}")
            col2.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
            col3.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.4f}")
            
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
            
            # Prediction plot
            st.subheader("📈 Actual vs Predicted")
            fig_reg = plot_regression_results(y_test, y_pred)
            st.pyplot(fig_reg)
            
            buf_reg = save_figure_high_quality(fig_reg, "regression_plot.png")
            st.download_button(
                label="📥 Download Regression Plot (600 DPI)",
                data=buf_reg,
                file_name="regression_plot_600dpi.png",
                mime="image/png"
            )
            
            # Feature Importance
            if hasattr(st.session_state['model'], 'feature_importances_'):
                st.subheader("📊 Feature Importance")
                fig_fi = plot_feature_importance(
                    st.session_state['model'], 
                    st.session_state['feature_names']
                )
                if fig_fi:
                    st.pyplot(fig_fi)
                    buf_fi = save_figure_high_quality(fig_fi, "feature_importance.png")
                    st.download_button(
                        label="📥 Download Feature Importance (600 DPI)",
                        data=buf_fi,
                        file_name="feature_importance_600dpi.png",
                        mime="image/png"
                    )
        
        elif problem_type == "clustering":
            labels = st.session_state['labels']
            X_scaled = st.session_state['X_scaled']
            
            # Metrics
            st.subheader("📊 Clustering Metrics")
            
            col1, col2 = st.columns(2)
            col1.metric("Silhouette Score", f"{silhouette_score(X_scaled, labels):.4f}")
            col2.metric("Davies-Bouldin Index", f"{davies_bouldin_score(X_scaled, labels):.4f}")
            
            # Cluster sizes
            st.subheader("📋 Cluster Distribution")
            cluster_counts = pd.Series(labels).value_counts().sort_index()
            st.bar_chart(cluster_counts)
            
            # Visualization
            st.subheader("🎨 Cluster Visualization")
            fig_cluster = plot_clustering_results(X_scaled, labels)
            st.pyplot(fig_cluster)
            
            buf_cluster = save_figure_high_quality(fig_cluster, "clustering.png")
            st.download_button(
                label="📥 Download Clustering Plot (600 DPI)",
                data=buf_cluster,
                file_name="clustering_600dpi.png",
                mime="image/png"
            )
    
    else:
        st.warning("⚠️ Please train a model first")

with tab4:
    st.header("Model Interpretability")
    
    if 'model' in st.session_state and st.session_state.get('problem_type') != 'clustering':
        st.info("🔍 **SHAP Analysis** - Understanding model predictions")
        
        st.markdown("""
        SHAP (SHapley Additive exPlanations) values show the contribution of each feature 
        to the model's predictions. This helps understand which features are most important 
        for individual predictions.
        """)
        
        if st.button("Generate SHAP Analysis"):
            st.warning("⚠️ SHAP analysis requires the `shap` library. Install it with: `pip install shap`")
            st.info("For now, feature importance plots are available in the Results tab.")
    
    else:
        st.warning("⚠️ Interpretability analysis is available after model training")

# Footer
st.markdown("---")
st.markdown("**AutoML Research Platform** | Built for researchers and data scientists | Publication-ready outputs")
