"""Document and package the isolated trial; never write to its parent project."""
from pathlib import Path
import hashlib,json,difflib,re,zipfile
import importlib.metadata as metadata
import fitz

ROOT=Path(__file__).resolve().parents[1]
assert ROOT.name.startswith('Backup_Trial_PartialMapping_')

NOTES='''轻量试改说明（2026-10-21）

一、位置与原稿保护
所有试改只发生在 Backup_Trial_PartialMapping_20261021 这个独立副本中。
父目录中的原稿没有覆盖。258 个原交付成员的 SHA-256 已逐项核对，均未变化。
baseline_snapshot/ 保存试改前的正文、补充材料、参考文献和章节/表格源码，供比较。
该快照不是另一个独立编译工程；试改工程以本目录的 main.tex 为入口。

二、改动范围
题目与现有章节结构保留。原稿正文中的 8 幅图、10 张表均保留，编号不变。
原有 41 个图表资产/表格数据文件逐字节一致，原有参考文献也完全一致。
没有改原语义公式、训练参数、温度、门限、生成器或约束解码器。
英文正文由 27 页变为 29 页；补充材料由 10 页变为 11 页。

新增位置：
- 第 3.7 节：Optional completion of a partially known class registry。
  仅在原框架外加一个可选步骤：某个接收机的部分词表映射未知时，
  用物理初始事件组比较候选映射，再用已映射语义重新关联。
- 第 4.8 节：Small probe with missing class correspondences。
  只新增表 11；原实验、原图表和原数值不替换。
- 补充材料第 S9 节：复用数据、锚点、候选枚举、适配/测试划分及边界。
- 摘要、引言、讨论和结论只做与新增小节相容的少量措辞调整。

三、为何没有沿用上次大改的方法
继续使用原稿的熵加权对数语义一致度。固定分组时，直接比较少量相容置换，
而不是替换为双线性语义特征，也没有引入新的分类评分器或大范围搜索架构。
四类中给一个锚点，余下三类只有 3!=6 种候选；公开小检验只有两种候选。
这样可以让缺失映射学习附着于原有框架，不改变原稿的基本研究定位。

四、已执行的小规模验证
受控检验只隐藏接收机 1 的三个类别对应，其他接收机映射已知。
复用原有数据和固定模型，1 个适配场景时恢复 26/30 个隐藏条目（86.7%），
事件 F1=0.7639；5 或 40 个适配场景时恢复全部条目，F1=0.7798，
与原稿已知映射结果一致。未利用数据的锚点相容补全为 F1=0.7026。
全部 800 个已知映射参考场景的 F1 与原稿逐项相同。

公开检验保留原来的六文件 LoED 实验。在这项额外检验中，单独把
2020-08-03 的 15,254 条记录作为适配数据，2020-09-02 的 8423 条作为测试。
只隐藏一个网关的 SF7/SF12 类别对应，另外四项及参考网关映射已知。
两个隐藏条目恢复正确，测试 F1 从未解映射时的 0.1270 恢复为 0.9961，
与原稿在这一天的完整映射结果一致。它不等于原稿两天合计的 F1=0.9975。

必须保留的边界：
- 这些公开未知词表来自实测元数据的受控编码隐藏，不是天然不同协议的词表。
- 受控适配描述符复用了原校准场景，原温度已经在这些场景上离线校准；
  不把它说成新采集的、完全独立的无标签验证集。公开检验的适配日期晚于原校准。
- 顺序映射与反馈方法在本次全部设置上持平。能支持“补全未知对应”，
  不能声称“反馈必然提高准确率”或“实测显著优于顺序方法”。
- 一对一映射、共有类别数、锚点及其他接收机映射仍然给定。

五、文件与复现
main.pdf：小幅试改正文。supplement.pdf：同步补充材料。
LIGHT_REVISION.patch：仅论文文字/新增小节/新表的变更对照。
results/partial_mapping_extension/：原始结果、学习映射、轨迹、预测和独立复核。
PARTIAL_MAPPING_PROTOCOL.txt：运行前写下的范围与设置。

在新环境安装 requirements-trial.txt 后，可从本副本目录运行：
python code/partial_mapping_extension.py
python code/verify_partial_mapping_extension.py
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex

code/apply_light_revision.py 是在干净副本上一次性应用文字增补的记录脚本，
不应在已修改稿上重复执行。通常只需编辑现有 TeX 文件并编译。
原 README 的其余内容属于继承的原稿说明；与本说明不一致时以本说明为准。
本副本尚未替换父目录原稿，也未执行投稿或对外发布。
'''

def main():
    (ROOT/'TRIAL_NOTES_ZH.txt').write_text(NOTES,encoding='utf-8')
    for filename in ['README.md','SUBMISSION_README.txt']:
        p=ROOT/filename;text=p.read_text(encoding='utf-8')
        notice='【独立轻量试改副本】当前只在本备份中增补部分未知映射小节；父目录原稿保持原样。改动、结果和复现步骤以 TRIAL_NOTES_ZH.txt 为准。\n\n'
        if not text.startswith('【独立轻量试改副本】'):p.write_text(notice+text,encoding='utf-8')
    names=['numpy','scipy','pandas','scikit-learn','threadpoolctl','PyMuPDF','Pillow']
    (ROOT/'requirements-trial.txt').write_text('# Trial verified on Python 3.12.14; original frozen parameters reused.\n'+'\n'.join(f'{n}=={metadata.version(n)}' for n in names)+'\n',encoding='utf-8')
    baseline=ROOT/'baseline_snapshot';patch=[]
    sources=[ROOT/'main.tex',ROOT/'supplement.tex',ROOT/'references.tex']+sorted((ROOT/'sections').glob('*.tex'))+sorted((ROOT/'tables').glob('*.tex'))
    changed=[]
    for path in sources:
        rel=path.relative_to(ROOT).as_posix();oldpath=baseline/rel
        old=oldpath.read_text(encoding='utf-8') if oldpath.exists() else ''
        new=path.read_text(encoding='utf-8')
        if old!=new:
            patch.extend(difflib.unified_diff(old.splitlines(keepends=True),new.splitlines(keepends=True),fromfile='original/'+rel,tofile='trial/'+rel))
            changed.append(rel)
    (ROOT/'LIGHT_REVISION.patch').write_text(''.join(patch),encoding='utf-8')
    caption_checks={}
    oldaux=ROOT.parent/'main.aux';newaux=ROOT/'main.aux'
    if oldaux.exists():
        parse=lambda p:dict(re.findall(r'\\newlabel\{([^}]+)\}\{\{([^}]*)\}',p.read_text(encoding='utf-8')))
        old=parse(oldaux);new=parse(newaux)
        for key,value in old.items():
            if '@' not in key and key.startswith(('fig:','tab:')):
                assert new[key]==value,(key,value,new.get(key));caption_checks[key]=value
    assert len([k for k in caption_checks if k.startswith('fig:')])==8
    assert len([k for k in caption_checks if k.startswith('tab:')])==10
    report=dict(status='PASS',modified_paper_sources=changed,original_figures=8,original_tables=10,
                original_caption_numbers_preserved=caption_checks,new_tables=1,
                main_pages=len(fitz.open(ROOT/'main.pdf')),supplement_pages=len(fitz.open(ROOT/'supplement.pdf')),
                visual_review='All main and supplement pages inspected in contact sheets; new method and result pages also inspected at larger resolution.')
    (ROOT/'trial_review/change_inventory.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    baseline_records=json.loads((ROOT/'BASELINE_FILE_HASHES.json').read_text())
    for r in baseline_records:assert hashlib.sha256((ROOT.parent/r['path']).read_bytes()).hexdigest()==r['sha256'],r['path']
    selected=set(ROOT/r['path'] for r in baseline_records)
    for pattern in ['sections/*partial_mapping.tex','tables/partial_mapping_probe.tex',
                    'code/*partial_mapping*.py','code/apply_light_revision.py','code/finalize_light_trial.py',
                    'results/partial_mapping_extension/*','baseline_snapshot/**/*',
                    'trial_review/change_inventory.json']:
        selected.update(p for p in ROOT.glob(pattern) if p.is_file())
    for name in ['TRIAL_NOTES_ZH.txt','PARTIAL_MAPPING_PROTOCOL.txt','LIGHT_REVISION.patch','BASELINE_FILE_HASHES.json',
                 'ORIGINAL_RELEASE_MANIFEST.json','requirements-trial.txt']:
        selected.add(ROOT/name)
    entries=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,
                  sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(selected)]
    manifest=dict(status='VERIFIED_LIGHT_TRIAL',original_directory_modified=False,files=entries)
    (ROOT/'TRIAL_FILE_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    archive=ROOT/'SupplementaryData_S1.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(selected):z.write(p,p.relative_to(ROOT).as_posix())
        z.write(ROOT/'TRIAL_FILE_MANIFEST.json','TRIAL_FILE_MANIFEST.json')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for entry in entries:assert hashlib.sha256(z.read(entry['path'])).hexdigest()==entry['sha256']
    check=dict(status='PASS',archive_bytes=archive.stat().st_size,
               archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),original_release_files_unchanged=len(baseline_records),
               package_members=len(entries)+1,backup_directory=str(ROOT))
    (ROOT/'trial_review/archive_check.json').write_text(json.dumps(check,indent=2),encoding='utf-8')
    print(json.dumps(check,indent=2))

if __name__=='__main__':main()
