import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

UP = '/kaggle/input/<your-dataset-folder-name>/'

train = pd.read_csv(UP + 'train.csv')
test = pd.read_csv(UP + 'test.csv')
att = pd.read_csv(UP + 'Attendance_series.csv')
notes = pd.read_csv(UP + 'Counsellor_notes.csv')

# ---------- Attendance feature engineering ----------
att_agg = att.groupby('student_id')['attendance_pct'].agg(
    att_mean='mean', att_std='std', att_min='min', att_max='max'
).reset_index()

# per-semester mean attendance
sem_pivot = att.groupby(['student_id', 'semester'])['attendance_pct'].mean().unstack()
sem_pivot.columns = [f'att_sem{c}_mean' for c in sem_pivot.columns]
sem_pivot = sem_pivot.reset_index()

# per-subject mean attendance
subj_pivot = att.groupby(['student_id', 'subject'])['attendance_pct'].mean().unstack()
subj_pivot.columns = [f'att_{c}_mean' for c in subj_pivot.columns]
subj_pivot = subj_pivot.reset_index()

# trend: slope of attendance over (semester, week) ordered index per student
# trend: slope of attendance over (semester, week) ordered index per student
# (vectorized closed-form linear regression slope, avoids slow .apply())
g = att.sort_values(['student_id', 'semester', 'week']).copy()
g['idx'] = g.groupby('student_id').cumcount()
g['xy'] = g['idx'] * g['attendance_pct']
g['xx'] = g['idx'] * g['idx']
agg2 = g.groupby('student_id').agg(
    n=('idx', 'count'),
    sum_x=('idx', 'sum'),
    sum_y=('attendance_pct', 'sum'),
    sum_xy=('xy', 'sum'),
    sum_xx=('xx', 'sum')
).reset_index()
denom = agg2['n'] * agg2['sum_xx'] - agg2['sum_x'] ** 2
slope = (agg2['n'] * agg2['sum_xy'] - agg2['sum_x'] * agg2['sum_y']) / denom.replace(0, np.nan)
att_trend = pd.DataFrame({'student_id': agg2['student_id'], 'att_trend': slope.fillna(0)})

# last semester (4) mean attendance specifically — recent behavior signal
last_sem = att[att['semester'] == 3].groupby('student_id')['attendance_pct'].mean().reset_index()
last_sem.columns = ['student_id', 'att_last_sem_mean']

att_feats = att_agg.merge(sem_pivot, on='student_id', how='left') \
                    .merge(subj_pivot, on='student_id', how='left') \
                    .merge(att_trend, on='student_id', how='left') \
                    .merge(last_sem, on='student_id', how='left')

# ---------- Counsellor notes feature engineering ----------
notes['note_lower'] = notes['counsellor_note'].str.lower()

keyword_map = {
    'note_stress': 'stress',
    'note_dropout_mention': 'dropping out',
    'note_demotivated': 'demotivated',
    'note_backlog': 'backlog',
    'note_struggling': 'struggling',
    'note_financial': 'financial',
    'note_tutoring': 'tutoring',
    'note_good': 'good',
    'note_no_action': 'no further action',
    'note_monitor': 'monitor',
}
for col, kw in keyword_map.items():
    notes[col] = notes['note_lower'].str.contains(kw, regex=False).astype(int)

# also label-encode the exact note text (small fixed vocabulary of 10 phrases)
le_note = LabelEncoder()
notes['note_id'] = le_note.fit_transform(notes['counsellor_note'])

note_feat_cols = ['student_id', 'note_id'] + list(keyword_map.keys())
notes_feats = notes[note_feat_cols]

# ---------- Merge everything ----------
def build(df):
    df = df.merge(att_feats, on='student_id', how='left')
    df = df.merge(notes_feats, on='student_id', how='left')
    return df

train_f = build(train)
test_f = build(test)

# ---------- Tabular feature engineering ----------
for df in [train_f, test_f]:
    cgpas = df[['cgpa_sem1', 'cgpa_sem2', 'cgpa_sem3', 'cgpa_sem4']]
    df['cgpa_mean'] = cgpas.mean(axis=1)
    df['cgpa_std'] = cgpas.std(axis=1)
    df['cgpa_trend'] = df['cgpa_sem4'] - df['cgpa_sem1']
    df['cgpa_min'] = cgpas.min(axis=1)
    backs = df[['backlogs_sem1', 'backlogs_sem2', 'backlogs_sem3']]
    df['backlogs_total'] = backs.sum(axis=1)
    df['backlogs_trend'] = df['backlogs_sem3'] - df['backlogs_sem1']

# ---------- Categorical encoding ----------
cat_cols = ['branch', 'gender', 'hostel_status', 'family_income', 'parent_education']
for col in cat_cols:
    train_f[col] = train_f[col].fillna('Missing')
    test_f[col] = test_f[col].fillna('Missing')
    le = LabelEncoder()
    combined = pd.concat([train_f[col], test_f[col]], axis=0)
    le.fit(combined)
    train_f[col] = le.transform(train_f[col])
    test_f[col] = le.transform(test_f[col])

# ---------- Fill numeric NaNs ----------
feature_cols = [c for c in train_f.columns if c not in ['student_id', 'dropout_risk']]
for col in feature_cols:
    if train_f[col].dtype in [np.float64, np.int64]:
        med = train_f[col].median()
        train_f[col] = train_f[col].fillna(med)
        test_f[col] = test_f[col].fillna(med)

X = train_f[feature_cols]
y = train_f['dropout_risk']
X_test = test_f[feature_cols]

print("Feature columns:", feature_cols)
print("X shape:", X.shape, "X_test shape:", X_test.shape)

# ---------- Cross-validated LightGBM ----------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros((len(X_test), 3))
oof_preds = np.zeros(len(X))
acc_scores = []

params = {
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_logloss',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'feature_fraction': 0.85,
    'bagging_fraction': 0.85,
    'bagging_freq': 1,
    'min_data_in_leaf': 20,
    'verbose': -1,
    'seed': 42,
}

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    train_set = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols + ['note_id'])
    val_set = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_cols + ['note_id'])

    model = lgb.train(
        params, train_set,
        num_boost_round=1000,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )

    val_pred = model.predict(X_val)
    oof_preds[val_idx] = np.argmax(val_pred, axis=1)
    acc = (oof_preds[val_idx] == y_val.values).mean()
    acc_scores.append(acc)
    print(f"Fold {fold}: accuracy = {acc:.4f}")

    test_preds += model.predict(X_test) / skf.n_splits

print("Mean OOF accuracy:", np.mean(acc_scores))
overall_acc = (oof_preds == y.values).mean()
print("Overall OOF accuracy:", overall_acc)

from sklearn.metrics import f1_score
macro_f1 = f1_score(y, oof_preds, average='macro')
print("Macro F1 score:", macro_f1)

final_preds = np.argmax(test_preds, axis=1)

# ---------- Build submission ----------
sample_sub = pd.read_csv(UP + 'sample_submission.csv')
submission = test_f[['student_id']].copy()
submission['dropout_risk'] = final_preds
# reorder to match sample submission order
submission = sample_sub[['student_id']].merge(submission, on='student_id', how='left')

assert submission.shape[0] == sample_sub.shape[0]
assert submission['dropout_risk'].isnull().sum() == 0

submission.to_csv('/kaggle/working/submission.csv', index=False)
print("Saved submission.csv")
print(submission['dropout_risk'].value_counts())
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def draw_box(ax, x, y, w, h, title, subtitle, facecolor):
    box = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.02,rounding_size=0.04",
                          linewidth=1, edgecolor='black', facecolor=facecolor)
    ax.add_patch(box)
    ax.text(x + w/2, y + h*0.62, title, ha='center', va='center',
            fontsize=10, fontweight='bold')
    ax.text(x + w/2, y + h*0.28, subtitle, ha='center', va='center',
            fontsize=8)

def draw_arrow(ax, x1, y1, x2, y2):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                             arrowstyle='-|>', mutation_scale=15,
                             linewidth=1, color='black')
    ax.add_patch(arrow)

# ---------------------------------------------------------------
# 1. WORKFLOW DIAGRAM
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title("Workflow: Multimodal Student Dropout Risk Prediction", fontsize=13, fontweight='bold', pad=15)

# Row 1: data sources
draw_box(ax, 0.3, 8, 2.8, 1.2, "Academic data\n(train.csv)", "Demographics, CGPA,\nbacklogs", "#cfe8ff")
draw_box(ax, 3.6, 8, 2.8, 1.2, "Attendance series", "Weekly % by\nsubject (LSTM-style)", "#cfe8ff")
draw_box(ax, 6.9, 8, 2.8, 1.2, "Counsellor notes", "Text observations\n(NLP)", "#cfe8ff")

# Row 2: feature engineering
draw_box(ax, 1.5, 6, 7, 1.2, "Feature engineering & merge", "45 features: CGPA trends, attendance\nstats/trend, note keywords + label-encoding", "#d4f4dd")

# Row 3: model
draw_box(ax, 2.5, 4, 5, 1.2, "LightGBM multiclass classifier", "5-fold stratified cross-validation\nwith early stopping", "#ffe3b3")

# Row 4: outputs
draw_box(ax, 0.3, 1.5, 2.8, 1.2, "Low risk (0)", "~62% of\npredictions", "#d4f4dd")
draw_box(ax, 3.6, 1.5, 2.8, 1.2, "Medium risk (1)", "~24% of\npredictions", "#ffe3b3")
draw_box(ax, 6.9, 1.5, 2.8, 1.2, "High risk (2)", "~14% of\npredictions", "#ffd1d1")

# Arrows row1 -> row2
draw_arrow(ax, 1.7, 8, 3, 7.2)
draw_arrow(ax, 5, 8, 5, 7.2)
draw_arrow(ax, 8.3, 8, 7, 7.2)

# Arrow row2 -> row3
draw_arrow(ax, 5, 6, 5, 5.2)

# Arrows row3 -> row4
draw_arrow(ax, 4.2, 4, 1.7, 2.7)
draw_arrow(ax, 5, 4, 5, 2.7)
draw_arrow(ax, 5.8, 4, 8.3, 2.7)

plt.tight_layout()
plt.savefig('/kaggle/working/workflow_diagram.png', dpi=150, bbox_inches='tight')
plt.show()

# ---------------------------------------------------------------
# 2. MODEL ARCHITECTURE / FEATURE GROUP DIAGRAM
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 7))
ax.set_xlim(0, 10)
ax.set_ylim(0, 9)
ax.axis('off')
ax.set_title("Model Architecture: Feature Groups -> LightGBM -> Risk Class", fontsize=13, fontweight='bold', pad=15)

# Feature group boxes (left column)
draw_box(ax, 0.3, 7.0, 4, 1.1, "Demographics & background", "branch, gender, hostel, family income,\nparent education, scholarship, job", "#cfe8ff")
draw_box(ax, 0.3, 5.6, 4, 1.1, "Academic performance", "CGPA sem1-4, CGPA mean/std/trend,\nbacklogs per semester, totals & trend", "#cfe8ff")
draw_box(ax, 0.3, 4.2, 4, 1.1, "Attendance signals", "mean/std/min/max, per-semester &\nper-subject means, trend slope", "#cfe8ff")
draw_box(ax, 0.3, 2.8, 4, 1.1, "Counsellor note signals", "note category (encoded) +\n10 keyword flags (stress, backlog, etc.)", "#cfe8ff")

# Model box (center-right)
draw_box(ax, 5.2, 4.7, 4, 1.6, "LightGBM\nGradient-Boosted Trees", "45-feature input\nmulti-class objective (3 classes)\n5-fold CV ensemble", "#ffe3b3")

# Output box
draw_box(ax, 5.7, 1.5, 3, 1.4, "Predicted dropout_risk", "0 = Low\n1 = Medium\n2 = High", "#d4f4dd")

# Arrows feature groups -> model
for yarrow in [7.5, 6.1, 4.7, 3.3]:
    draw_arrow(ax, 4.3, yarrow, 5.2, 5.5)

# Arrow model -> output
draw_arrow(ax, 7.2, 4.7, 7.2, 2.9)

plt.tight_layout()
plt.savefig('/kaggle/working/architecture_diagram.png', dpi=150, bbox_inches='tight')
plt.show()

print("Saved workflow_diagram.png and architecture_diagram.png to /kaggle/working/")

# ---------------------------------------------------------------
# 3. CONFUSION MATRIX (out-of-fold predictions)
# ---------------------------------------------------------------
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y, oof_preds)
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm, cmap='Blues')
for i in range(3):
    for j in range(3):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='black')
ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1, 2])
ax.set_xticklabels(['Low', 'Medium', 'High'])
ax.set_yticklabels(['Low', 'Medium', 'High'])
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix (OOF)')
plt.colorbar(im)
plt.tight_layout()
plt.savefig('/kaggle/working/confusion_matrix.png', dpi=150)
plt.show()

# ---------------------------------------------------------------
# 4. FEATURE IMPORTANCE (last fold's model)
# ---------------------------------------------------------------
imp = model.feature_importance(importance_type='gain')
feat_imp = pd.Series(imp, index=feature_cols).sort_values(ascending=True).tail(15)
fig, ax = plt.subplots(figsize=(7, 5))
feat_imp.plot(kind='barh', ax=ax, color='steelblue')
ax.set_title('Top 15 Feature Importances (Gain)')
plt.tight_layout()
plt.savefig('/kaggle/working/feature_importance.png', dpi=150)
plt.show()

print("Saved confusion_matrix.png and feature_importance.png to /kaggle/working/")
