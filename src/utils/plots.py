import matplotlib.pyplot as plt
from collections import defaultdict

def plot_training_history(history, output_dir="./"):
    """
    Plots:
    1) Train loss vs Eval loss (per epoch)
    2) Epoch vs Eval QWK
    """

    # ---- Aggregate train loss per epoch ----
    train_loss_per_epoch = defaultdict(list)

    eval_epochs = []
    eval_losses = []
    eval_qwks = []

    for entry in history:
        # Train step logs
        if "loss" in entry and "epoch" in entry:
            epoch = int(entry["epoch"])
            train_loss_per_epoch[epoch].append(entry["loss"])

        # Eval logs (appear once per epoch)
        if "eval_loss" in entry:
            eval_epochs.append(entry["epoch"])
            eval_losses.append(entry["eval_loss"])
            eval_qwks.append(entry["eval_qwk"])

    # Average train loss per epoch
    train_epochs = sorted(train_loss_per_epoch.keys())
    train_losses = [
        sum(train_loss_per_epoch[e]) / len(train_loss_per_epoch[e])
        for e in train_epochs
    ]

    # ================= Plot 1: Train Loss vs Eval Loss =================
    plt.figure(figsize=(10, 6))
    plt.plot(train_epochs, train_losses, marker="o", linewidth=2, label="Train Loss")
    plt.plot(eval_epochs, eval_losses, marker="s", linewidth=2, label="Eval Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train Loss vs Eval Loss")
    plt.legend()
    plt.grid(alpha=0.3)

    loss_path = f"{output_dir}/train_vs_eval_loss.png"
    plt.savefig(loss_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    # ================= Plot 2: Epoch vs Eval QWK =================
    plt.figure(figsize=(10, 6))
    plt.plot(eval_epochs, eval_qwks, marker="^", linewidth=2, color="tab:blue")

    plt.xlabel("Epoch")
    plt.ylabel("QWK Score")
    plt.title("Eval QWK Over Epochs")
    plt.grid(alpha=0.3)

    qwk_path = f"{output_dir}/eval_qwk.png"
    plt.savefig(qwk_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    print(f"Saved:\n- {loss_path}\n- {qwk_path}")


# history = [{'loss': 1.9723, 'grad_norm': 32.19810104370117, 'learning_rate': 1.886836027713626e-05, 'epoch': 0.057770075101097634, 'step': 50}, {'loss': 0.5467, 'grad_norm': 8.000741958618164, 'learning_rate': 1.771362586605081e-05, 'epoch': 0.11554015020219527, 'step': 100}, {'loss': 0.5459, 'grad_norm': 17.568754196166992, 'learning_rate': 1.655889145496536e-05, 'epoch': 0.1733102253032929, 'step': 150}, {'loss': 0.4664, 'grad_norm': 9.925410270690918, 'learning_rate': 1.5404157043879908e-05, 'epoch': 0.23108030040439054, 'step': 200}, {'loss': 0.4724, 'grad_norm': 3.862006902694702, 'learning_rate': 1.4249422632794459e-05, 'epoch': 0.28885037550548814, 'step': 250}, {'loss': 0.4912, 'grad_norm': 6.483991622924805, 'learning_rate': 1.309468822170901e-05, 'epoch': 0.3466204506065858, 'step': 300}, {'loss': 0.4513, 'grad_norm': 15.378613471984863, 'learning_rate': 1.1939953810623557e-05, 'epoch': 0.4043905257076834, 'step': 350}, {'loss': 0.4218, 'grad_norm': 6.103455543518066, 'learning_rate': 1.0785219399538108e-05, 'epoch': 0.4621606008087811, 'step': 400}, {'loss': 0.4084, 'grad_norm': 6.679830074310303, 'learning_rate': 9.630484988452657e-06, 'epoch': 0.5199306759098787, 'step': 450}, {'loss': 0.3987, 'grad_norm': 5.817183494567871, 'learning_rate': 8.475750577367207e-06, 'epoch': 0.5777007510109763, 'step': 500}, {'loss': 0.403, 'grad_norm': 9.324394226074219, 'learning_rate': 7.321016166281756e-06, 'epoch': 0.635470826112074, 'step': 550}, {'loss': 0.4053, 'grad_norm': 5.277344226837158, 'learning_rate': 6.166281755196305e-06, 'epoch': 0.6932409012131716, 'step': 600}, {'loss': 0.4131, 'grad_norm': 9.664894104003906, 'learning_rate': 5.0115473441108554e-06, 'epoch': 0.7510109763142692, 'step': 650}, {'loss': 0.3614, 'grad_norm': 4.368160247802734, 'learning_rate': 3.8568129330254045e-06, 'epoch': 0.8087810514153668, 'step': 700}, {'loss': 0.4062, 'grad_norm': 5.448825359344482, 'learning_rate': 2.7020785219399544e-06, 'epoch': 0.8665511265164645, 'step': 750}, {'loss': 0.3448, 'grad_norm': 6.23114538192749, 'learning_rate': 1.5473441108545037e-06, 'epoch': 0.924321201617521, 'step': 800}, {'loss': 0.3911, 'grad_norm': 6.745013236999512, 'learning_rate': 3.926096997690532e-07, 'epoch': 0.9820912767186597, 'step': 850}, {'eval_loss': 0.39271169900894165, 'eval_qwk': 0.7673636269320633, 'eval_runtime': 41.7645, 'eval_samples_per_second': 82.893, 'eval_steps_per_second': 10.368, 'epoch': 1.0, 'step': 866}, {'train_runtime': 697.0239, 'train_samples_per_second': 19.863, 'train_steps_per_second': 1.242, 'total_flos': 0.0, 'train_loss': 0.5215954510774679, 'epoch': 1.0, 'step': 866}] 
# plot_training_metrics(history, output_dir="./")