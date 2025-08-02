from lec_utils import *
from ipywidgets import interact

def show_decision_boundary(model, X_train, y_train, title=''):
    from plotly.subplots import make_subplots

    # Create grid for decision boundary
    tol = 0
    x_min, x_max = X_train['Glucose'].min() - tol, X_train['Glucose'].max() + tol
    y_min, y_max = X_train['BMI'].min() - tol, X_train['BMI'].max() + tol
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400),
                        np.linspace(y_min, y_max, 400))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Create figure
    fig = make_subplots()
    
    # Add decision boundary
    fig.add_trace(go.Contour(
        x=np.linspace(x_min, x_max, 400),
        y=np.linspace(y_min, y_max, 400),
        z=Z,
        colorscale=[(0, 'orange'), (1, 'blue')],
        opacity=0.5,
        showscale=False
    ))

    # Add scatter points
    fig.add_trace(go.Scatter(
        x=X_train.loc[y_train == 0, 'Glucose'],
        y=X_train.loc[y_train == 0, 'BMI'],
        mode='markers',
        marker=dict(color='orange', size=8),
        name='no diabetes'
    ))
    
    fig.add_trace(go.Scatter(
        x=X_train.loc[y_train == 1, 'Glucose'],
        y=X_train.loc[y_train == 1, 'BMI'],
        mode='markers',
        marker=dict(color='blue', size=8),
        name='yes diabetes'
    ))

    # Update layout
    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        xaxis_title='Glucose',
        yaxis_title='BMI',
        showlegend=True,
        legend=dict(font=dict(size=12)),
        width=700,
        height=500
    )

    fig.update_xaxes(range=[x_min, x_max])
    fig.update_yaxes(range=[y_min, y_max])

    return fig

def show_one_feature_plot(X_train, y_train):
    fig = px.scatter(X_train.assign(tumor_status_int=y_train,
                                    tumor_status=y_train.astype(str).replace({'0': 'Benign', '1': 'Malignant'})),
           x='mean radius',
           y='tumor_status_int',
           color='tumor_status',
           color_discrete_map={'Benign': 'orange', 'Malignant': 'blue'},
           width=800, size=np.ones(X_train.shape[0]), size_max=8)
    fig.update_layout(showlegend=False, xaxis_title='Mean Radius', yaxis_title='Tumor Status')
    return fig

from sklearn.linear_model import LogisticRegression

def show_one_feature_plot_with_logistic(X_train, y_train):
    model_logistic = LogisticRegression()
    model_logistic.fit(X_train[['mean radius']].to_numpy(), y_train)

    x_min, x_max = X_train['mean radius'].min(), X_train['mean radius'].max()

    x = np.linspace(x_min, x_max)
    y = model_logistic.predict_proba(x.reshape(-1, 1))[:, 1]

    fig = show_one_feature_plot(X_train, y_train)
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        name='Logistic Regression Model',
        line=dict(color='#097054', width=4)
    ))
    
    return fig

def show_one_feature_plot_with_logistic_and_y_threshold(X_train, y_train, T):
    model_logistic = LogisticRegression()
    model_logistic.fit(X_train[['Glucose']].to_numpy(), y_train)

    x = np.linspace(0, 250)
    y = model_logistic.predict_proba(x.reshape(-1, 1))[:, 1]

    fig = show_one_feature_plot(X_train, y_train)
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        name='Logistic Regression Model',
        line=dict(color='#097054', width=4)
    ))

    fig.add_trace(go.Scatter(
        x=[x[0] - 15, x[-1] + 15],
        y=[T, T],
        name=f'Threshold of T = {T}',
        line=dict(color='purple', width=4)
    ))

    fig.update_xaxes(range=(x[0], x[-1]))
    
    return fig

def show_one_feature_plot_with_logistic_and_x_threshold(X_train, y_train, T):
    model_logistic = LogisticRegression()
    model_logistic.fit(X_train[['Glucose']].to_numpy(), y_train)

    x = np.linspace(0, 250)
    y = model_logistic.predict_proba(x.reshape(-1, 1))[:, 1]

    fig = show_one_feature_plot(X_train, y_train)
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        name='Logistic Regression Model',
        line=dict(color='#097054', width=4)
    ))

    w0, w1 = model_logistic.intercept_[0], model_logistic.coef_[0][0]

    T_ops = (np.log(T / (1 - T)) - w0) / w1

    fig.add_trace(go.Scatter(
        x=[T_ops, T_ops],
        y=[0, 1],
        name=f'Threshold of T = {T}',
        line=dict(color='purple', width=4)
    ))

    fig.update_xaxes(range=(x[0], x[-1]))
    
    return fig

def show_one_feature_plot_in_1D(X_train, y_train, thres=True):

    fig = px.scatter(X_train.assign(Diabetes=y_train, 
                          Outcome=y_train.astype(str).replace({'0': 'no diabetes', '1': 'yes diabetes'})),
           x='Glucose',
           y=[0] * X_train.shape[0],
           color='Outcome',
           color_discrete_map={'no diabetes': 'orange', 'yes diabetes': 'blue'},
           size_max=10,
           size=[5] * X_train.shape[0],
           width=1000)
    
    if thres:
        fig.add_trace(go.Scatter(
            x=[139.17, 139.17],
            y=[-0.1, 0.1],
            name=f'Threshold of Glucose = 140',
            line=(dict(color='purple', width=4))
        ))

    fig.update_yaxes(range=(-0.03, 0.03))

    fig.add_annotation(
        x=170,
        y=0.015,
        text="<span style='color:blue'>classified as <b>diabetes</b> ➡️</span>",
        showarrow=False
    )

    fig.add_annotation(
        x=100,
        y=0.015,
        text="<span style='color:orange'>⬅️ classified as <b>no diabetes</b></span>",
        showarrow=False
    )

    return fig

def make_prop_plot(X_train, y_train):
    col = 'Glucose'
    vals = X_train[col]
    bins = np.linspace(vals.min() - 3, vals.max() + 3, 12)
    fig = (
        pd.cut(X_train[col], bins)
        .to_frame()
        .assign(Outcome=y_train)
        .groupby(col, observed=True)
        .agg(['mean', 'size'])
        .reset_index()
        .pipe(lambda df: df.assign(left=df[col].apply(lambda x: int(x.left))))
        .pipe(lambda df: px.scatter(
            x=df['left'], y=df['Outcome']['mean'], size=np.ones(df.shape[0]) * 3, size_max=12,
#             color=df['Outcome']['mean']
        ))
    #     .plot(kind='scatter', x='left', y='mean')
        .update_xaxes(range=(vals.min() - 40, vals.max() + 20))
        .update_layout(width=800, 
                       xaxis_title='Glucose', 
                       yaxis_title='Proportion of Individuals with Diabetes',
                       coloraxis_showscale=False)
    )
    
    return fig

def plot_sigmoid(w0, w1):
    xs = np.linspace(-10, 10, 10000)
    inps = w0 + w1 * xs
    ys = 1 / (1 + np.e ** (-inps))
    
    title = r'$y = \sigma(' + (f'{w0}' if w0 != 0 else '') + ('+' if w1 >= 0 else '-') + (f'{np.abs(w1)}x' if w1 != 1 else 'x') + r')$'
    title = fr'w0 = {w0}, w1 = {w1}'

    fig = px.line(x=xs, y=ys, title=title)
    fig.update_traces(line=dict(width=4))
    
    return fig

def show_three_sigmoids():
    fig1 = plot_sigmoid(w0=0, w1=1)
    fig2 = plot_sigmoid(w0=15, w1=5)
    fig3 = plot_sigmoid(w0=-0.5, w1=-0.4)

    combined_fig = make_subplots(rows=1, cols=3,
                                 subplot_titles = [figi.layout.title['text'] for figi in [fig1, fig2, fig3]])

    combined_fig.add_trace(fig1.data[0], row=1, col=1)
    combined_fig.add_trace(fig2.data[0], row=1, col=2)
    combined_fig.add_trace(fig3.data[0], row=1, col=3)

    combined_fig.update_layout(width=1200)
    
    return combined_fig

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve
from sklearn.linear_model import LogisticRegression

def predict_thresholded(X_train, y_train, X, T):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    probs = model.predict_proba(X)[:, 1]
    return (probs >= T).astype(int)

def plot_vs_threshold(X_train, y_train, metric):

    fn_dict = {'Precision': precision_score, 'Recall': recall_score, 'Accuracy': accuracy_score, 'F1 Score': f1_score}

    metric_fn = fn_dict[metric]

    thresholds = np.arange(0, 1.005, 0.005)
    values = []
    for t in thresholds:
        preds = predict_thresholded(X_train, y_train, X_train, t)
        value = metric_fn(y_train, preds)
        values.append(value)
        
    fig = px.line(x=thresholds, y=values,
                  title=f'{metric} vs. Threshold',
                  labels={'x': 'Threshold', 'y': f'Training {metric}'})
    
    return fig.update_layout(width=800)

from sklearn.metrics import confusion_matrix

def show_confusion(X_train, y_train, T):
    # Get predictions and confusion matrix
    y_pred = predict_thresholded(X_train, y_train, X_train, T)
    cm = confusion_matrix(y_train, y_pred)
    
    # Create plotly heatmap
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted Negative', 'Predicted Positive'],
        y=['Actual Negative', 'Actual Positive'],
        colorscale='Blues',
        text=[['True Negatives (TN)', 'False Positives (FP)'],
              ['False Negatives (FN)', 'True Positives (TP)']],
        texttemplate='%{text}<br>%{z}',
        textfont=dict(size=12),
        hovertemplate='Count: %{z}<br>Category: %{text}'
    ))

    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Confusion Matrix<br>with Threshold {T}',
            y=0.95,  # Adjust title position
            x=0.5,
            xanchor='center',
            yanchor='top'
        ),
        width=600,
        height=500,
        margin=dict(t=90),  # Add top margin
        xaxis=dict(side='top'),
        yaxis=dict(autorange='reversed')
    )

    return fig

def pr_curve(X_train, y_train):

    precisions = []
    recalls = []

    thresholds = np.arange(0, 1, 0.005)
    values = []
    for t in thresholds:
        preds = predict_thresholded(X_train, y_train, X_train, t)
        precision = precision_score(y_train, preds)
        recall = recall_score(y_train, preds)
        precisions.append(precision)
        recalls.append(recall)
        
    fig = px.line(x=recalls, y=precisions, hover_name='Threshold = ' + pd.Series(thresholds).astype(str),
                  title=f'Precision vs. Recall',
                  labels={'x': 'Recall', 'y': f'Precision'})
    
    return fig.update_layout(width=800)


def draw_roc_curve(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_train)[:, 1]

    fprs, tprs, thresholds = roc_curve(y_train.to_numpy(), probs)
        
    fig = px.line(x=fprs, y=tprs, hover_name='Threshold = ' + pd.Series(thresholds).astype(str),
                  title=f'ROC Curve<br>(True Positive Rate vs. False Positive Rate)',
                  labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'})
    
    return fig.update_layout(width=800)

def show_balancing_demo():
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Create imbalanced dataset: 12 class 0, 3 class 1
    np.random.seed(42)

    # Feature values - higher values slightly more likely to be class 1
    X = np.array([2.5, 2.5, 2.5, 2.1, 1.8, 2.3, 1.9, 2.0, 1.7, 2.2, 1.6, 2.4, 1.5, 2.0, 1.8, 3.2, 3.5, 3.1]).reshape(-1, 1)

    # Target: severe imbalance (12 zeros, 3 ones)
    y = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1])

    # Create DataFrame for visualization
    df = pd.DataFrame({
        'feature': X.flatten(),
        'target': y
    })

    # Model 1: Default LogisticRegression (no class balancing)
    lr_default = LogisticRegression(random_state=42)
    lr_default.fit(X, y)

    # Predictions
    y_pred_default = lr_default.predict(X)
    y_proba_default = lr_default.predict_proba(X)

    # Model 2: Balanced LogisticRegression
    lr_balanced = LogisticRegression(class_weight='balanced', random_state=42)
    lr_balanced.fit(X, y)

    # Predictions
    y_pred_balanced = lr_balanced.predict(X)
    y_proba_balanced = lr_balanced.predict_proba(X)

    # Show the class weights used
    class_weights = lr_balanced.class_weight
    if class_weights == 'balanced':
        # Calculate what sklearn computes internally
        n_samples = len(y)
        n_classes = len(np.unique(y))
        class_0_weight = n_samples / (n_classes * sum(y == 0))
        class_1_weight = n_samples / (n_classes * sum(y == 1))

    # Visualization with Plotly
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Dataset Distribution', 'Default Model', 'Balanced Model'),
        horizontal_spacing=0.1
    )

    # Plot 1: Data distribution
    colors_class = ['blue', 'red']
    for class_val in [0, 1]:
        mask = y == class_val
        fig.add_trace(
            go.Scatter(
                x=X[mask].flatten(),
                y=[class_val] * sum(mask),
                mode='markers',
                marker=dict(size=12, color=colors_class[class_val], opacity=0.7),
                name=f'Class {class_val} (n={sum(mask)})',
                showlegend=True if class_val == 0 else True
            ),
            row=1, col=1
        )

    # Plot 2: Default model probabilities
    fig.add_trace(
        go.Scatter(
            x=X.flatten(),
            y=y_proba_default[:, 1],
            mode='markers',
            marker=dict(size=12, color='green', opacity=0.7),
            name='Default Probabilities',
            showlegend=False
        ),
        row=1, col=2
    )

    # Add decision threshold line for default model
    fig.add_hline(
        y=0.5, 
        line=dict(color='red', dash='dash', width=2),
        row=1, col=2
    )

    # Plot 3: Balanced model probabilities
    fig.add_trace(
        go.Scatter(
            x=X.flatten(),
            y=y_proba_balanced[:, 1],
            mode='markers',
            marker=dict(size=12, color='orange', opacity=0.7),
            name='Balanced Probabilities',
            showlegend=False
        ),
        row=1, col=3
    )

    # Add decision threshold line for balanced model
    fig.add_hline(
        y=0.5,
        line=dict(color='red', dash='dash', width=2),
        row=1, col=3
    )

    # Update layout
    fig.update_xaxes(title_text='Feature Value', row=1, col=1)
    fig.update_xaxes(title_text='Feature Value', row=1, col=2)
    fig.update_xaxes(title_text='Feature Value', row=1, col=3)

    fig.update_yaxes(title_text='Class', row=1, col=1, range=[-0.1, 1.1])
    fig.update_yaxes(title_text='Predicted Probability', row=1, col=2, range=[-0.1, 1.1])
    fig.update_yaxes(title_text='Predicted Probability', row=1, col=3, range=[-0.1, 1.1])

    fig.update_layout(
        height=400,
        width=1200,
        showlegend=True
    )

    fig.show()