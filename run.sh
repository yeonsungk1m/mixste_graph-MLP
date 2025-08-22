
# baseline
python run.py -k gt -f 243 -s 243 -l log/run -c checkpoint/243-gt -gpu 4

# seal
python run-seal.py -k gt -f 243 -s 243 -l log/run -c checkpoint/243-cpn-gt-seal13 -gpu 4 \
	--lr_loss 1e-4 --energy_weight 1e-5 --em_loss_type margin --margin_type mpjpe