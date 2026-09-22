"""Read saved Backdoor results and build a Traditional Chinese stage report."""

import argparse
from datetime import date
import hashlib
import json
from pathlib import Path
import statistics
import zipfile

from build_science_report import (
    ROOT,
    Cm,
    Document,
    Inches,
    OxmlElement,
    Pt,
    RGBColor,
    font_manager,
    np,
    plt,
    qn,
    readcsv,
)

ALPHAS = (0.01, 0.1, 1, 10, 100)
SEEDS = (42, 43, 44)
METRICS = ("clean_accuracy_percent", "asr_percent", "clean_loss")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(study):
    """Check saved evidence and independently recompute final statistics."""
    checked = {}
    for manifest in sorted(study.rglob("checksums.json")):
        for name, expected in read_json(manifest).items():
            path = manifest.parent / name
            actual = sha256(path)
            assert actual == expected, f"Checksum mismatch: {path}"
            checked[str(path.relative_to(ROOT))] = actual
    records = []
    curves = {}
    summaries = {}
    dataset_hashes = None
    for alpha in ALPHAS:
        folder = study / f"alpha_{alpha:g}"
        settings = read_json(folder / "config.json")
        assert settings["alpha"] == alpha
        for key, expected in {
            "poison_ratio": 0.2,
            "target_label": 0,
            "trigger_size": 3,
            "trigger_row": 25,
            "trigger_column": 25,
            "trigger_value": 1,
            "seeds": list(SEEDS),
            "num_clients": 5,
            "attacker": 0,
            "rounds": 10,
            "local_epochs": 1,
            "batch_size": 32,
            "learning_rate": 0.01,
            "optimizer": "SGD",
        }.items():
            assert settings[key] == expected, (folder, key)
        hashes = read_json(folder / "dataset_hashes.json")
        if dataset_hashes is None:
            dataset_hashes = hashes
        assert hashes == dataset_hashes
        environment = read_json(folder / "environment.json")
        for name, digest in environment["source_sha256"].items():
            assert sha256(folder / "source" / name) == digest
        summaries[alpha] = {}
        for seed in SEEDS:
            configs = []
            pair = {}
            for mode in ("clean", "backdoor"):
                run = folder / f"seed_{seed}" / mode
                cfg = read_json(run / "config.json")
                assert cfg["alpha"] == alpha and cfg["seed"] == seed
                assert cfg["condition"] == mode
                assert cfg["client_sample_counts"] == [12000] * 5
                assert cfg["asr_sample_count"] == 9020
                selected = cfg["poisoned_local_positions"]
                assert len(set(selected)) == (
                    2400 if mode == "backdoor" else 0
                )
                assert all(0 <= i < 12000 for i in selected)
                configs.append(cfg)
                with np.load(run / "partitions.npz") as saved:
                    parts = [saved[f"client_{cid}"] for cid in range(5)]
                    assert np.array_equal(
                        np.sort(np.concatenate(parts)), np.arange(60000)
                    )
                    for cid, part in enumerate(parts):
                        assert len(part) == 12000
                        assert hashlib.sha256(part.tobytes()).hexdigest() == (
                            cfg["partition_sha256"][cid]
                        )
                rows = readcsv(run / "results.csv")
                assert [int(row["round"]) for row in rows] == list(
                    range(1, 11)
                )
                parsed = [{m: float(row[m]) for m in METRICS} for row in rows]
                assert all(
                    np.isfinite(list(row.values())).all() for row in parsed
                )
                assert all(
                    0 <= row["clean_accuracy_percent"] <= 100
                    and 0 <= row["asr_percent"] <= 100
                    and row["clean_loss"] >= 0
                    for row in parsed
                )
                curves[alpha, seed, mode] = parsed
                pair[mode] = parsed[-1]
            for key in (
                "initial_parameters_sha256",
                "partition_sha256",
                "shuffle_seeds",
                "client_sample_counts",
            ):
                assert configs[0][key] == configs[1][key], (alpha, seed, key)
            records.append(
                {
                    "alpha": alpha,
                    "seed": seed,
                    **pair,
                    "difference": {
                        m: pair["backdoor"][m] - pair["clean"][m]
                        for m in METRICS
                    },
                }
            )
        subset = [r for r in records if r["alpha"] == alpha]
        for mode in ("clean", "backdoor", "difference"):
            summaries[alpha][mode] = {
                metric: (
                    statistics.mean(r[mode][metric] for r in subset),
                    statistics.stdev(r[mode][metric] for r in subset),
                )
                for metric in METRICS
            }
        saved_summary = read_json(folder / "summary.json")
        for saved in saved_summary["final_round_per_seed"]:
            actual = next(r for r in subset if r["seed"] == saved["seed"])
            for metric in METRICS:
                assert saved[metric] == actual[saved["condition"]][metric]
    mapping = {
        "clean_accuracy_percent": ("clean", METRICS[0]),
        "backdoor_clean_accuracy_percent": ("backdoor", METRICS[0]),
        "clean_asr_percent": ("clean", METRICS[1]),
        "backdoor_asr_percent": ("backdoor", METRICS[1]),
        "paired_clean_accuracy_change_pp": ("difference", METRICS[0]),
        "paired_asr_increase_pp": ("difference", METRICS[1]),
    }
    saved_rows = readcsv(study / "per_seed.csv")
    assert len(saved_rows) == 15
    for record in records:
        matches = [
            r
            for r in saved_rows
            if float(r["alpha"]) == record["alpha"]
            and int(r["seed"]) == record["seed"]
        ]
        assert len(matches) == 1
        for field, (mode, metric) in mapping.items():
            assert np.isclose(float(matches[0][field]), record[mode][metric])
    saved_stats = readcsv(study / "summary.csv")
    assert len(saved_stats) == 5
    for alpha in ALPHAS:
        matches = [r for r in saved_stats if float(r["alpha"]) == alpha]
        assert len(matches) == 1
        for field, (mode, metric) in mapping.items():
            for index, suffix in enumerate(("mean", "sample_sd")):
                assert np.isclose(
                    float(matches[0][f"{field}_{suffix}"]),
                    summaries[alpha][mode][metric][index],
                )
    return records, summaries, curves, checked


def make_figures(out, records, summaries, curves):
    fontfile = Path("/mnt/c/Windows/Fonts/msjh.ttc")
    if fontfile.exists():
        font_manager.fontManager.addfont(str(fontfile))
        fontname = font_manager.FontProperties(fname=str(fontfile)).get_name()
    else:
        fontname = "Noto Sans CJK TC"
    plt.rcParams.update(
        {
            "font.family": fontname,
            "axes.unicode_minus": False,
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    def save(fig, name):
        fig.savefig(out / f"{name}.png", dpi=180, bbox_inches="tight")
        fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)

    x = np.arange(5)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, metric, title in zip(
        axes, METRICS, ("正常測試準確率", "攻擊成功率 ASR")
    ):
        for mode, label, offset in (
            ("clean", "Clean", -0.07),
            ("backdoor", "Backdoor", 0.07),
        ):
            values = [summaries[a][mode][metric] for a in ALPHAS]
            ax.errorbar(
                x + offset,
                [v[0] for v in values],
                yerr=[v[1] for v in values],
                marker="o",
                capsize=4,
                label=label,
            )
        ax.set_xticks(x, [str(a) for a in ALPHAS])
        ax.set_xlabel("Alpha（各設定等距排列）")
        ax.set_ylabel(f"{title}（%）")
        ax.grid(alpha=0.2)
        ax.legend()
    axes[0].set_ylim(80, 94)
    axes[1].set_ylim(0, 105)
    save(fig, "overview")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, metric, label in zip(
        axes, METRICS, ("正常準確率變化", "ASR 增加量")
    ):
        for seed in SEEDS:
            values = [
                next(
                    r for r in records if r["alpha"] == a and r["seed"] == seed
                )["difference"][metric]
                for a in ALPHAS
            ]
            ax.plot(x, values, "o-", label=f"Seed {seed}")
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_xticks(x, [str(a) for a in ALPHAS])
        ax.set_xlabel("Alpha（各設定等距排列）")
        ax.set_ylabel(f"{label}（百分點）")
        ax.grid(alpha=0.2)
        ax.legend()
    save(fig, "paired_seeds")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for alpha in ALPHAS:
        for ax, metric in zip(axes, METRICS):
            values = np.array(
                [
                    [r[metric] for r in curves[alpha, seed, "backdoor"]]
                    for seed in SEEDS
                ]
            )
            ax.plot(range(1, 11), values.mean(axis=0), label=f"α = {alpha}")
    for ax, label in zip(
        axes,
        (
            "正常準確率（%）",
            "攻擊成功率 ASR（%）",
        ),
    ):
        ax.set_xlabel("聯邦訓練輪數")
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9)
    save(fig, "rounds")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for mode in ("clean", "backdoor"):
        values = [summaries[a][mode]["clean_loss"] for a in ALPHAS]
        axes[0].errorbar(
            x,
            [v[0] for v in values],
            yerr=[v[1] for v in values],
            marker="o",
            capsize=4,
            label=mode.title(),
        )
    values = [summaries[a]["difference"]["clean_loss"] for a in ALPHAS]
    axes[1].errorbar(
        x,
        [v[0] for v in values],
        yerr=[v[1] for v in values],
        marker="o",
        capsize=4,
        color="#b95433",
    )
    for ax, label in zip(axes, ("正常測試 Loss", "有攻擊與無攻擊的 Loss 差")):
        ax.set_xticks(x, [str(a) for a in ALPHAS])
        ax.set_xlabel("Alpha（各設定等距排列）")
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
    axes[0].legend()
    axes[1].axhline(0, color="gray", linestyle="--")
    save(fig, "loss")


def build_document(out, study, records, summaries):
    # Reuse the existing report's page layout, styles, and footer conventions.
    doc = Document(ROOT / "reports/current/聯邦學習_目前研究報告.docx")
    for element in list(doc.element.body):
        if element.tag != qn("w:sectPr"):
            doc.element.body.remove(element)
    p = doc.add_paragraph

    def page(title):
        doc.add_page_break()
        doc.add_heading(title, 1)

    def sub(title):
        doc.add_heading(title, 2)

    def table(headers, rows):
        tb = doc.add_table(rows=1, cols=len(headers))
        tb.style = "Light Shading Accent 1"
        for cell, value in zip(tb.rows[0].cells, headers):
            cell.text = str(value)
        tb.rows[0]._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))
        for row in rows:
            for cell, value in zip(tb.add_row().cells, row):
                cell.text = str(value)
        for row in tb.rows:
            row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    paragraph.paragraph_format.space_after = Pt(3)
                    for run in paragraph.runs:
                        run.font.size = Pt(9)

    def picture(name, caption):
        doc.add_picture(
            str(out / "figures" / f"{name}.png"), width=Inches(6.5)
        )
        p(caption, "Caption")

    def value(alpha, mode, metric, digits=2):
        mean, sd = summaries[alpha][mode][metric]
        return f"{mean:.{digits}f} ± {sd:.{digits}f}"

    p("科學展覽研究報告｜Backdoor 階段成果")
    doc.add_paragraph(
        "聯邦學習中的後門攻擊\n資料分配不同，攻擊效果會改變嗎？", "Title"
    )
    doc.add_paragraph("以 MNIST 手寫數字比較有攻擊與無攻擊的結果", "Subtitle")
    p("科別：電腦科學與資訊工程科")
    p("關鍵詞：聯邦學習、後門攻擊、資料投毒、攻擊成功率、Non-IID")
    p(f"整理日期：{date.today().isoformat()}")
    p("學校／作者／指導老師／參賽編號：待填")
    p(
        "本報告接續前一份 Model Poisoning（模型投毒）報告，整理目前完成的 Backdoor（後門攻擊）實驗。我們比較五組 Alpha 設定，每組做三次有攻擊與無攻擊的對照實驗，共有 30 次訓練結果。其中 Alpha＝0.1 使用前期測試的結果，不重複計算。"
    )
    table(
        ["階段", "本次完成內容"],
        [
            ["前期測試", "Alpha＝0.1；確認圖樣、對照方式與資料都正確"],
            [
                "比較不同 Alpha",
                "0.01、0.1、1、10、100；固定修改攻擊端 20% 圖片",
            ],
            ["報告", "核對原始資料，整理平均、標準差、差值與圖表"],
            ["尚未包含", "防禦、其他圖樣設定、更多 seeds 與其他模型／資料集"],
        ],
    )
    p(
        "這份報告使用已完成的實驗資料，不需要重新訓練模型。先前的 Model Poisoning 報告與原始實驗結果也都保留。"
    )

    page("摘要")
    p(
        "本研究想了解：如果聯邦學習中的一個參與端，在部分訓練圖片上加上固定圖樣，並把答案改成同一個數字，模型會不會學到錯誤的規則？我們使用 MNIST 手寫數字資料，讓五個參與端一起訓練模型。每端都有 12,000 張圖片，合併模型時各占 20%，避免資料量不同影響比較。"
    )
    p(
        "我們把 client 0 設為攻擊者，選出它 20% 的圖片，也就是 2,400 張，在右下角加上 3×3 的白色方塊，並把訓練答案設為 0。這個固定圖樣稱為「觸發圖樣」。實驗比較 Alpha＝0.01、0.1、1、10、100，每組使用隨機種子（seed）42、43、44，分別做無攻擊（Clean）與有攻擊（Backdoor）的十輪訓練。除了測量正常圖片的辨識準確率，我們也測量攻擊成功率（ASR）：原本不是 0 的圖片，加上圖樣後，有多少被模型判成 0。"
    )
    p(
        "第十輪結果顯示，五組有攻擊模型的平均 ASR 為 60.69%～67.82%，無攻擊模型則為 0.78%～0.95%。不過，有攻擊模型在正常圖片上的平均準確率，只下降了 0.03～0.25 個百分點。Alpha＝0.01 的 ASR 為 67.82% ± 28.08%，三次結果差很多；Alpha＝100 則為 61.86% ± 0.56%，三次結果較接近。這表示，只看正常圖片的準確率，可能看不出這次實驗中的後門問題。但目前還不能說 Alpha 越大或越小，攻擊就一定越容易成功。"
    )
    sub("Abstract")
    p(
        "This study asks whether a model can learn a wrong rule from a small set of changed training images. We use MNIST and five clients, each with 12,000 images. Client 0 adds a white 3×3 square to the bottom-right corner of 20% of its images and sets their training labels to 0. We compare five alpha settings. Each setting uses three random seeds, with and without the attack, for ten rounds. The final average attack success rate (ASR) is 60.69%–67.82% for the attacked models. Their normal test accuracy drops by only 0.03–0.25 percentage points on average. Results vary widely across seeds at alpha 0.01. Our results show why we need to check both normal accuracy and ASR. They do not show that a larger or smaller alpha always makes the attack stronger."
    )

    page("壹、前言")
    sub("一、研究動機")
    p(
        "前一階段，我們讓攻擊者把模型更新放大，觀察整體準確率會下降多少。這一階段，我們想知道：如果模型平常還能正常辨識，卻會把帶有特定圖樣的圖片判成指定數字，只看準確率是否足夠？聯邦學習會合併各端的訓練結果，形成共同模型［1］。後門攻擊則可能讓模型在遇到特定圖樣時，做出攻擊者想要的錯誤判斷［2］。我們沿用原本的 MNIST 聯邦學習程式，測試這種情況。"
    )
    sub("二、研究目的")
    for text in (
        "建立一套能依照相同設定重做的後門實驗，先使用一個攻擊者。",
        "讓每端的圖片數量和合併模型時的比例相同，再比較不同 Alpha 的結果。",
        "同時查看正常準確率、Loss 和 ASR，避免只看一個數值就下結論。",
        "先比較同一個 seed 下有攻擊與無攻擊的差別，再看三次結果是否接近。",
    ):
        p("• " + text)
    sub("三、與前一階段的區別")
    table(
        ["項目", "先前 Model Poisoning", "本次 Backdoor"],
        [
            [
                "做法",
                "放大 client 0 的模型更新",
                "修改 client 0 部分訓練圖片及標籤",
            ],
            [
                "主要觀察",
                "正常辨識能力下降",
                "加上圖樣後，被判成指定數字的比例",
            ],
            [
                "攻擊設定",
                "更新放大倍率 Scale",
                "圖片修改比例、觸發圖樣、指定答案",
            ],
            ["合併時的比例", "原 A/B 依各端實際樣本數", "本次每端固定 20%"],
        ],
    )
    p(
        "這張表比較的是本專案使用的兩種攻擊做法。兩個階段的資料分配方式和觀察重點不同，因此不能直接拿結果判斷哪一種攻擊比較強。其他後門攻擊也可能使用不同做法，不一定只修改圖片和答案。"
    )

    page("貳、研究設備及器材")
    env = read_json(study / "alpha_0.1/environment.json")
    table(
        ["項目", "設定／用途"],
        [
            ["資料", "MNIST；訓練 60,000 張、測試 10,000 張；28×28 灰階"],
            ["模型", "784 個輸入 → 128 個隱藏節點 → ReLU → 10 類輸出"],
            ["執行", "CPU 單執行緒，依序模擬五端；設定相同時盡量得到一致結果"],
            ["模型合併", "FedAvg：依各端圖片數量加權平均；每輪五端都參加"],
            ["Python", env["python"]],
            *[[name, version] for name, version in env["packages"].items()],
            ["報告工具", "Matplotlib 圖表、python-docx 文件；沿用原報告格式"],
        ],
    )
    p(
        "表中的版本來自實驗時保存的 environment.json。沒有記錄的硬體型號和執行時間不另外填入。這次是在同一台電腦上依序模擬五個參與端，因此無法用來判斷真實網路傳送資料需要多少時間。"
    )
    sub("資料來源與實驗範圍")
    p(
        "本報告使用 results/Backdoor_test/alpha_sweep/ 中的原始結果、設定檔與資料分配紀錄。Alpha＝0.1 沿用 mnist_alpha_0.1_ratio_0.2 的前期測試，另外四組則使用後續完成的結果。總共是五組 × 三個 seeds × 兩種條件＝30 次訓練，每次十輪，共 300 輪。前期測試的六次已經包含在內。"
    )
    p(
        "整理報告時，我們從原始 CSV 重新計算平均值和標準差，再與原本的整理表比對，確認數值一致。這個過程不會重新訓練模型。"
    )

    page("參、研究過程及方法")
    table(
        ["條件", "設定"],
        [
            ["變因 Alpha", "0.01、0.1、1、10、100；只改變這一項設定"],
            ["Seeds", "42、43、44；每個編號都做有攻擊與無攻擊的比較"],
            ["每端資料量", "12,000 張；用 Dirichlet 分配，並固定每端張數"],
            ["訓練", "10 輪；每輪各端資料訓練一遍；每批 32 張"],
            ["訓練方法", "SGD；學習率 0.01；不放大模型更新"],
            [
                "攻擊者／修改圖片",
                "client 0；固定 2,400 張，占攻擊端 20%、全部 4%",
            ],
            ["觸發／目標", "右下角 3×3 白色；像素值 1.0；目標數字 0"],
            ["位置", "列、欄都從 0 編號；圖樣位在第 25～27 列與欄"],
            ["公平比較", "兩組使用相同起始模型、分配圖片及排列順序"],
        ],
    )
    p(
        "Seed（隨機種子）是控制程式隨機選擇的編號，會影響起始模型、圖片分配和排列順序。我們使用 42、43、44 做三次比較；同一個 seed 的有攻擊與無攻擊組保持相同起點，編號較大不代表攻擊較強。"
    )
    sub("一、怎麼分配資料、選擇要修改的圖片")
    p(
        "我們使用 Dirichlet 方法，決定各類數字分到不同參與端的比例，再加上「每端只能分到 12,000 張」的限制。Alpha 是控制分配差異的參數，較小時通常會讓各端拿到的數字種類比例更不一樣。但因為這次固定了圖片數量，分配方式與前一階段不同。每端實際拿到多少張 0、1、2 等數字，都有另外保存。"
    )
    p(
        "每次實驗用 seed＋2000 決定要修改哪些圖片，從攻擊端選出 20%，也就是 2,400 張，而且不重複抽取。選到的圖片如果原本就是 0，仍會加上圖樣，但答案維持 0。所以「修改 2,400 張圖片」不代表 2,400 張的答案全部都有改變。十輪訓練都使用同一批選中的圖片。換成不同 Alpha 後，資料分配會改變，因此即使選到相同的本地編號，也不一定是同一張原始圖片。"
    )
    sub("二、每輪流程")
    p(
        "把共同模型交給五個參與端 → 各端用自己的資料訓練一遍，攻擊端在讀取選中的圖片時加上圖樣，並把答案設為 0 → 各端交回訓練後的模型 → 每端各占 20%，合併成新模型 → 分別測試正常圖片與加上圖樣的圖片。"
    )
    p(
        "程式會先複製圖片再加上圖樣，不會改掉原始圖片。其他四個參與端，以及無攻擊組，都使用原本的圖片和答案。正常測試圖片也不修改。訓練圖片的排列順序由 seed＋1000＋client_id 決定，讓同一組有攻擊與無攻擊實驗的順序一致。"
    )

    page("參、研究過程及方法（續）")
    sub("三、怎麼判斷攻擊效果")
    p(
        "正常準確率（%）＝正常測試圖片中，答對的張數 ÷ 10,000 × 100。有攻擊和無攻擊的模型，都使用同一份未修改的測試圖片。"
    )
    p(
        "ASR（%）＝原本不是 0 的圖片，加上圖樣後被判成 0 的張數 ÷ 9,020 × 100。測試集原本有 980 張數字 0，這些圖片不列入計算，因為把它們判成 0 本來就是正確答案。加上圖樣的測試圖片另外使用，不會取代正常測試圖片。"
    )
    p(
        "無攻擊模型也要測試加上圖樣的圖片，才能知道它原本就有多大機會把這些圖片判成 0。Loss 則用來看模型的答案離正確答案有多遠，本研究使用交叉熵計算。程式會依每批圖片的張數計算平均，讓每張正常測試圖片的影響相同。"
    )
    sub("四、怎麼比較兩組結果")
    p(
        "我們把同一個 seed 的兩次訓練放在一起比較，稱為「配對比較」。差值一律用有攻擊減去無攻擊：ΔAccuracy＝Backdoor 正常準確率 − Clean 正常準確率；ΔASR＝Backdoor ASR − Clean ASR；ΔLoss＝Backdoor 正常 Loss − Clean 正常 Loss。符號 Δ 代表前後的差。"
    )
    p(
        "準確率與 ASR 的差使用「百分點」，例如 90% 降到 89%，就是下降 1 個百分點。ΔAccuracy 小於 0 表示正常準確率下降；ΔASR 大於 0 表示更多觸發圖片被判成 0；ΔLoss 大於 0 表示正常測試的錯誤程度增加。如果某次準確率略微提高，也保留原本數值。這裡的相減方向與舊報告的「準確率損害＝Clean − Poisoned」相反，閱讀時要注意。"
    )
    p(
        "表格中的「平均 ± 樣本標準差」，用來表示三次結果的平均值和分散程度。標準差越大，代表三次結果差得越多。這裡用樣本標準差，計算時除以 n−1＝2。配對差也是先算出每個 seed 的差，再算三個差值的平均與標準差。主要比較第 10 輪，不能把十輪訓練當成十次獨立實驗。圖上的誤差棒表示標準差，不是信賴區間。"
    )
    sub("五、資料核對")
    p(
        "寫報告前，我們先確認資料是否完整，包括每次都有十輪結果、每端圖片數量正確、資料分配沒有遺漏或重複，以及有攻擊和無攻擊組是否使用相同的起始模型、圖片與排列設定。我們也用 SHA-256 雜湊值比對檔案；可以把它想成檔案的指紋，用來檢查內容是否改變。接著重算第十輪結果，與整理表比對。核對紀錄保存在 validation.json。這些檢查是確認保存資料的一致性，並不是重新訓練一次。"
    )

    page("肆、研究結果｜正常準確率與 ASR")
    table(
        ["Alpha", "Clean 準確率 %", "Backdoor 準確率 %", "ΔAccuracy 百分點"],
        [
            [
                f"{a:g}",
                value(a, "clean", METRICS[0]),
                value(a, "backdoor", METRICS[0]),
                value(a, "difference", METRICS[0]),
            ]
            for a in ALPHAS
        ],
    )
    table(
        ["Alpha", "Clean ASR %", "Backdoor ASR %", "ΔASR 百分點"],
        [
            [
                f"{a:g}",
                value(a, "clean", METRICS[1]),
                value(a, "backdoor", METRICS[1]),
                value(a, "difference", METRICS[1]),
            ]
            for a in ALPHAS
        ],
    )
    p(
        "表中都是第十輪結果，列出三個 seeds 的平均值和樣本標準差，並四捨五入到小數點後兩位。"
    )
    picture(
        "overview",
        "圖 1　不同 Alpha 的正常準確率和 ASR。誤差棒表示三次結果的樣本標準差。左右兩圖的縱軸範圍不同，左圖縮小範圍，方便看出準確率的差別。",
    )
    p(
        "五組正常準確率平均只下降 0.03～0.25 個百分點，但 ASR 卻平均增加 59.87～66.87 個百分點。也就是說，模型對正常圖片的表現幾乎沒變，遇到加上圖樣的圖片時，卻更容易把答案判成 0。"
    )

    page("肆、研究結果｜各 seed 與訓練過程")
    picture(
        "paired_seeds",
        "圖 2　三個 seeds 各自的差值，都是有攻擊減去無攻擊。連線是為了方便比較五組設定，不代表 Alpha 之間的其他數值也會照著線變化。",
    )
    p(
        "Alpha＝0.01 時，seeds 42、43、44 的有攻擊 ASR 分別是 88.20%、35.79%、79.46%。雖然平均值較高，但其中一次只有 35.79%，所以不能說每次攻擊都很成功。Alpha＝100 的三次 ASR 是 62.46%、61.36%、61.75%，這三次結果就比較接近。"
    )
    picture(
        "rounds",
        "圖 3　有攻擊組在十輪訓練中的平均變化，每條線都是三個 seeds 的平均。各次實驗差多少，仍要搭配圖 2 和附錄查看。",
    )
    p(
        "從第十輪的平均 ASR 來看，Alpha 增加時，ASR 並沒有一直增加或一直下降。圖中的曲線只能說明這十輪的結果，還不能推測繼續訓練，或停止加入攻擊圖片後，會發生什麼事。"
    )

    page("肆、研究結果｜正常測試 Loss")
    table(
        ["Alpha", "Clean Loss", "Backdoor Loss", "ΔLoss"],
        [
            [
                f"{a:g}",
                value(a, "clean", METRICS[2], 4),
                value(a, "backdoor", METRICS[2], 4),
                value(a, "difference", METRICS[2], 4),
            ]
            for a in ALPHAS
        ],
    )
    p(
        "Loss 是用未修改的正常測試圖片計算，通常越低越好。表格同樣列出三次結果的平均值和樣本標準差；ΔLoss 則先比較每個 seed 的有攻擊與無攻擊結果，再整理成平均值。"
    )
    picture(
        "loss",
        "圖 4　正常圖片的 Loss。左圖比較有攻擊與無攻擊，右圖是兩者的差。誤差棒表示三次結果的樣本標準差。",
    )
    p(
        "準確率只看模型最後有沒有答對，Loss 還會受到模型對正確答案的把握程度影響。因此，準確率和 Loss 不一定一起變好或變差。但正常圖片的 Loss 仍看不出加上圖樣後的情況，所以還需要 ASR。"
    )

    page("伍、討論與研究限制")
    sub("一、為何正常準確率幾乎不變，仍需要注意？")
    p(
        "這五組實驗中，有攻擊模型在正常圖片上的準確率，與無攻擊模型很接近；但遇到加上圖樣的圖片時，被判成 0 的比例卻高很多。如果只測正常圖片，就看不到這個問題。不過，這不代表攻擊一定無法被發現，因為本階段還沒有測試偵測方法。"
    )
    sub("二、固定樣本數解決了什麼？")
    p(
        "每端都固定 12,000 張圖片，合併模型時就各占 20%。這樣比較不同 Alpha 時，攻擊者不會因為拿到較多圖片，就在合併時占比較大的比例。不過，Alpha 還是會改變每端拿到哪些數字，以及哪些圖片被選來加上圖樣。所以目前還不能確定，這些因素各自影響 ASR 多少。"
    )
    sub("三、為什麼 Alpha＝0.01 不能只看平均？")
    p(
        "三次實驗中，一次 ASR 約 35.79%，另外兩次約 79.46% 和 88.20%。如果只寫平均 67.82%，就不容易看出這個差別。接下來可以增加實驗次數，也可以分別看攻擊端有哪些數字、哪些數字較容易被判成 0。只靠這三次結果，還不能說資料越不平均，後門攻擊就一定越強或越弱。"
    )
    sub("四、目前限制")
    for text in (
        "只有三個 seeds，實驗次數還不多，也還沒用統計檢定判斷差異是否明顯。",
        "只測試 MNIST、一種小型全連接模型、十輪訓練，以及固定的攻擊端。",
        "只使用一種 3×3 白色圖樣、固定位置、目標數字 0，以及 20% 的攻擊端圖片。",
        "每輪都使用同一批加上圖樣的圖片，還沒測試停止攻擊後，模型是否仍會判錯。",
        "本報告沒有納入防禦、Sybil、多個攻擊者或真實網路環境的實驗結果。",
        "這次固定每端圖片數量，與原本 A/B 實驗的分配方法不同，不能把兩階段的結果差異都當成攻擊方式造成的。",
    ):
        p("• " + text)

    page("陸、結論與後續工作")
    sub("一、目前結論")
    p(
        "這次實驗讓五個參與端擁有一樣多的圖片，並設定一個攻擊者。攻擊者在自己的 20% 訓練圖片上加入白色方塊，並把答案設為 0。五組 Alpha 的結果都顯示，這個做法會讓模型更容易把帶有圖樣的圖片判成 0。平均 ASR 約 60.69%～67.82%，但正常圖片的準確率平均下降不超過 0.25 個百分點。"
    )
    p(
        "因此，判斷模型是否受到後門攻擊影響時，不能只看正常準確率，也要一起看 Loss 和 ASR。另外，Alpha＝0.01 的三次結果差很多，提醒我們不能只做一次實驗，或只看平均值就下結論。目前也不能說 Alpha 越大或越小，ASR 就一定越高。"
    )
    sub("二、後續可以研究的方向")
    table(
        ["優先工作", "想知道什麼、怎麼比較"],
        [
            ["增加 seeds", "先不改其他設定，多做幾次，看看結果是否接近"],
            [
                "分別查看各種數字的結果",
                "看看哪些數字更容易被判成 0；原本的 0 不列入 ASR",
            ],
            [
                "比較圖片修改比例",
                "其他設定不變，也加入完全不修改圖片的 0% 對照組",
            ],
            [
                "更換攻擊端、答案或圖樣",
                "每次只改一項，看看換個設定是否仍有相同結果",
            ],
            [
                "停止攻擊與防禦方法",
                "先看停止修改圖片後的結果，再測防禦能否降低 ASR",
            ],
        ],
    )
    sub("三、研究進度")
    p(
        "本報告整理的工作包括前期測試、五組 Alpha 比較、原始資料核對與圖表。更多實驗次數、其他圖片修改比例、其他模型與資料集，以及停止攻擊後的影響和防禦效果，都留待後續研究。這些後續結果不包含在本份報告中。"
    )

    page("柒、參考資料與附錄")
    p(
        "［1］McMahan, B. et al.（2017）. Communication-Efficient Learning of Deep Networks from Decentralized Data. PMLR 54, 1273–1282. https://proceedings.mlr.press/v54/mcmahan17a.html （模型平均與聯邦學習背景。）"
    )
    p(
        "［2］Gu, T., Dolan-Gavitt, B., & Garg, S. BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain. arXiv:1708.06733. https://arxiv.org/abs/1708.06733 （觸發式後門背景；本報告不是該文實驗的完整重現。）"
    )
    p(
        "［3］本專案保存的 Backdoor 原始結果、來源快照、設定與彙總表：results/Backdoor_test/alpha_sweep/；pilot：results/Backdoor_test/mnist_alpha_0.1_ratio_0.2/。"
    )
    p(
        "［4］本專案 Model Poisoning 階段報告：reports/current/聯邦學習_目前研究報告.docx；僅沿用文件樣式與章節安排，舊版研究進度不代表本階段狀態。"
    )
    sub("附錄一：各 seed 第十輪結果")
    table(
        ["α", "Seed", "正常準確率 C/B %", "ASR C/B %", "ΔAcc／ΔASR"],
        [
            [
                f"{r['alpha']:g}",
                r["seed"],
                f"{r['clean'][METRICS[0]]:.2f} / {r['backdoor'][METRICS[0]]:.2f}",
                f"{r['clean'][METRICS[1]]:.2f} / {r['backdoor'][METRICS[1]]:.2f}",
                f"{r['difference'][METRICS[0]]:.2f} / {r['difference'][METRICS[1]]:.2f}",
            ]
            for r in records
        ],
    )
    p(
        "C 是無攻擊組，B 是有攻擊組；差值都是 B−C，單位是百分點。完整數值與各 seed 的 Loss，可查看同資料夾的 final_metrics.json。"
    )
    sub("附錄二：如何重新產生報告")
    p(
        "在專案根目錄執行：.venv/bin/python scripts/build_backdoor_report.py --output reports/backdoor_rebuilt"
    )
    p(
        "這個指令只整理已保存的結果，不會開始訓練。輸出包括 Word、四張 PNG／PDF 圖表、統計數值與核對紀錄。報告的 PDF 閱讀版由 Word 經 LibreOffice 轉出。"
    )
    path = out / "聯邦學習_Backdoor階段報告.docx"
    doc.save(path)
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
    assert len(Document(path).inline_shapes) == 4
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "reports/backdoor"
    )
    args = parser.parse_args()
    study = ROOT / "results/Backdoor_test/alpha_sweep"
    records, summaries, curves, checked = validate(study)
    args.output.mkdir(parents=True, exist_ok=False)
    figures = args.output / "figures"
    figures.mkdir()
    make_figures(figures, records, summaries, curves)
    path = build_document(args.output, study, records, summaries)
    for name, data in {
        "final_metrics.json": {"per_seed": records, "summary": summaries},
        "validation.json": {
            "verified": True,
            "settings": 5,
            "matched_pairs": 15,
            "condition_runs": 30,
            "metric_rows": 300,
            "checks": [
                "saved SHA-256 manifests",
                "partition coverage and hashes",
                "matched initialization hashes and shuffle seeds",
                "fixed sample counts and attack settings",
                "raw metrics versus per-seed and summary CSVs",
            ],
            "training_started": False,
            "input_sha256": checked,
            "report_script_sha256": sha256(Path(__file__)),
        },
    }.items():
        (args.output / name).write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    (args.output / "README.md").write_text(
        "# Backdoor 階段報告\n\n"
        "- 聯邦學習_Backdoor階段報告.docx：可編輯報告。\n"
        "- figures/：四張 PNG 圖與 PDF 向量圖。\n"
        "- final_metrics.json：各 seed 結果與重算統計。\n"
        "- validation.json：資料核對與輸入雜湊。\n\n"
        "重建（使用新目錄）：`.venv/bin/python scripts/build_backdoor_report.py "
        "--output reports/backdoor_rebuilt`。不會啟動訓練。\n",
        encoding="utf-8",
    )
    print(f"Created {path}; verified 30 runs / 300 metric rows.")


if __name__ == "__main__":
    main()
