# Paper v1：基于导师原稿的逐节修订

`Physics-Informed NoProp.tex` 复制自 `初始/Physics-Informed NoProp.tex`。
创建副本时，两者的 SHA-256 均为：

`A07F62A5E506D33CCDA8D4CE828DD8791634717576041A3B7FF31004D8457865`

工作稿目前只增加了修订标注所需的 LaTeX 宏，没有修改标题、摘要、正文、
公式、表格或参考文献。导师原稿 `初始/` 始终保持只读，作为永久对照基线。

## 修订规则

1. 按 Introduction、Related Work、Methodology、Experiments、Conclusion 的顺序进行。
2. 每次只修改已经确认的段落，不跨章节批量重写。
3. 执行“非错误不修改”的最小修订原则：不因语言风格、表达偏好或论证方式
   改写导师原文，只修正与当前数据、代码、实验结果或数学实现冲突的事实。
4. 经确认的事实修正使用 `\rev{...}`，在 PDF 中显示浅黄色底色。
5. 删除内容通过 Git diff 和导师原稿核对，不直接改动 `初始/`。
6. 每完成一轮都单独编译并检查引用、公式和版面，再进入下一轮。

编译命令：

```powershell
cd paper-v1
latexmk -pdf -outdir=out -interaction=nonstopmode -halt-on-error "Physics-Informed NoProp.tex"
```
