from train_stage.main_stage import MainStage


class PretrainStage(MainStage):
    def compute_batch_loss(self, batch):
        src = batch.src

        output, _, _ = self.model.encoder(src[:-1])  # turn off teacher forcing

        # trg = [trg sent len, batch size]
        # output = [trg sent len, batch size, output dim]

        output = output[1:].view(-1, output.shape[-1])
        trg = src[2:].view(-1)

        # trg = [(trg sent len - 1) * batch size]
        # output = [(trg sent len - 1) * batch size, output dim]

        loss = self.criterion(output, trg)
        return loss
