# 数值算子测评与 Agent 接入说明

## 1. 算子测评套件（operator_validation.py）

与实验 A.3 / A.7 对齐，对任意「数值列变换算子」做三类检查：

- **sanity**：均值/方差保持、Δ 界、未变比例（满足设定容差则 pass）
- **reconstruction**：线性回归攻击（用扰动后 Y 预测原始 X），R² 低于阈值则视为「难以还原」通过
- **multi_run**：同一输入多次运行，成对 KS / ρ，用于观察算子随机性（不设硬性 pass）

## 2. 接入新算子后如何自动验证

### 方式一：通过 Agent（推荐）

```python
from skills_and_agent import build_default_registry, PrivacyAgent

registry = build_default_registry()
agent = PrivacyAgent(registry)

# 接入新算子后
# registry.register(Skill(id="my_op", ..., target="numeric", fn=my_fn, ...))

report = agent.validate_numeric_skills(x=None, n_synthetic=2000)
print(report["all_pass"])   # True/False
print(report["summary"])    # 各 skill_id: PASS / FAIL
# report["skills"] 含每个算子的详细 checks（sanity / reconstruction / multi_run）
```

- `x=None` 时使用长度为 `n_synthetic` 的合成列，无需真实数据即可自检。
- 若提供真实列 `x`，可传入：`agent.validate_numeric_skills(x=your_column)`。

### 方式二：直接调用测评函数

```python
from skills_and_agent import validate_registry_numeric, validate_numeric_operator

# 对整表数值 Skill 测评
report = validate_registry_numeric(registry, x=None)

# 对单个算子测评（fn 签名为 (x, config) -> y）
result = validate_numeric_operator(my_fn, x_ref, config={"max_diff": 1.0})
print(result["pass"], result["checks"])
```

## 3. 阈值与可选参数

- `sanity_tol_mean` / `sanity_tol_var`：sanity 通过容差（默认 1e-6 / 1e-4）。
- `recon_r2_fail_threshold`：重建 R² 低于此值才认为「难以还原」通过（默认 0.98）。
- `recon_use_mlp=True`：同时跑 MLP 重建（需安装 scikit-learn）。
- `multi_run_K`：多次运行次数（默认 3）。
- `checks`：可只跑部分，如 `("sanity", "reconstruction")`。

详见 `operator_validation.validate_numeric_operator` 的文档与参数。

## 4. 与离线实验的关系

- 完整 A.3～A.7 实验（MIMIC 数据、多变量、多参数）在 `` 下按 A2～A7 脚本跑。
- 本测评套件用于 **Agent 侧快速自检**：用户新增数值算子后，无需跑全量实验即可判断是否满足基本变换需求（均值/方差、可重建性、随机性等）。
