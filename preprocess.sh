#!/bin/bash
PPATH=$(dirname $(readlink -f "$0"))
DPATH=${PPATH}
CELL=$1
resolution=$2
juicer_tool=$3
ratio=16

mkdir -p "$DPATH/$CELL"
mkdir -p "$DPATH/$CELL/intra_NONE"
mkdir -p "$DPATH/$CELL/intra_VC"
mkdir -p "$DPATH/$CELL/intra_KR"
mkdir -p "$DPATH/$CELL/intra_HR"
mkdir -p "$DPATH/$CELL/intra_LR"

#merge raw data
find "$DPATH/$CELL" -name "*_merged_nodups.txt.gz"|xargs zcat | sort -k3,3d -k7,7d > "$DPATH/$CELL/total_merged_nodups.txt"

#downsample
num=$(cat $DPATH/$CELL/total_merged_nodups.txt |wc -l)
num_downsample=`expr $(($num/$ratio))`
shuf -n $num_downsample $DPATH/$CELL/total_merged_nodups.txt | sort -k3,3d -k7,7d  > $DPATH/$CELL/total_merged_nodups_downsample_ratio_$ratio.txt
echo "merge data done!"

#generate .HIC file using juicer tool, -xmx50g indicates 50g for memory which can be replaced with an appropriate value
echo "java -Xmx50g  -jar $juicer_tool pre $DPATH/$CELL/total_merged_nodups.txt $DPATH/$CELL/total_merged.hic hg19"
java -Xmx50g  -jar $juicer_tool pre $DPATH/$CELL/total_merged_nodups.txt $DPATH/$CELL/total_merged.hic hg19
java -Xmx50g  -jar $juicer_tool pre $DPATH/$CELL/total_merged_nodups_downsample_ratio_$ratio.txt $DPATH/$CELL/total_merged_downsample_ratio_$ratio.hic hg19

chromes=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 "X" "Y")

#generate Hi-C raw contacts using juicer tool
for chrom in ${chromes[@]}
do
	java -jar $juicer_tool dump observed NONE $DPATH/$CELL/total_merged.hic $chrom $chrom BP $resolution $DPATH/$CELL/intra_NONE/chr$chrom_10k_intra_NONE.txt 
	java -jar $juicer_tool dump observed VC $DPATH/$CELL/total_merged.hic $chrom $chrom BP $resolution $DPATH/$CELL/intra_VC/chr$chrom_10k_intra_VC.txt 
	java -jar $juicer_tool dump observed KR $DPATH/$CELL/total_merged.hic $chrom $chrom BP $resolution $DPATH/$CELL/intra_KR/chr$chrom_10k_intra_KR.txt 
	java -jar $juicer_tool dump observed NONE $DPATH/$CELL/total_merged.hic $chrom $chrom BP $resolution $DPATH/$CELL/intra_HR/HR_10k_NONE.chr$chrom -d

	java -jar $juicer_tool dump observed NONE $DPATH/$CELL/total_merged_downsample_ratio_$ratio.hic $chrom $chrom BP $resolution $DPATH/$CELL/intra_NONE/chr$chrom_10k_intra_NONE_downsample_ratio$ratio.txt
	java -jar $juicer_tool dump observed VC $DPATH/$CELL/total_merged_downsample_ratio_$ratio.hic $chrom $chrom BP $resolution $DPATH/$CELL/intra_VC/chr$chrom_10k_intra_VC_downsample_ratio$ratio.txt
	java -jar $juicer_tool dump observed KR $DPATH/$CELL/total_merged_downsample_ratio_$ratio.hic $chrom $chrom BP $resolution $DPATH/$CELL/intra_KR/chr$chrom_10k_intra_KR_downsample_ratio$ratio.txt
	java -jar $juicer_tool dump observed NONE $DPATH/$CELL/total_merged_downsample_ratio_$ratio.hic $chrom $chrom BP $resolution $DPATH/$CELL/intra_LR/LR_10k_NONE.chr$chrom -d
done

