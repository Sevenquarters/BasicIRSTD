import os
from graphviz import Digraph

def draw_chinese_architecture_comparison_vector_pdf():
    # 创建全局定向前向图
    dot = Digraph('Architecture_Comparison_CN_Vector', comment='GCA and DSPG Evolution Chinese Vector')
    
    # 设置大图属性：横向布局，纯白背景
    dot.attr(rankdir='LR', bgcolor='white')
    
    # 强制显式设置字符集为 UTF-8
    dot.attr(charset='UTF-8')
    
    # =========================================================================
    # 【终极完美无损修复】
    # 完美支持中文 PDF 的字体设置，全部锁定为微软雅黑 (Microsoft YaHei)
    # =========================================================================
    dot.attr('graph', fontname='Microsoft YaHei', fontsize='14') # 强行管住子图标题 (a) (b) (c)
    dot.attr('node', fontname='Microsoft YaHei', fontsize='12', style='filled,rounded')  # 管住所有方框
    dot.attr('edge', fontname='Microsoft YaHei', fontsize='10')  # 管住所有线条

    # =========================================================================
    # 子图 1：基线跳跃连接
    # =========================================================================
    with dot.subgraph(name='cluster_baseline') as c:
        c.attr(label='(a) 基线连接 (Baseline)', color='gray', bgcolor='#FAFAFA', style='dashed,rounded')
        
        c.node('b_enc', '编码器特征\n[B, C, H, W]\n(空间特征)', shape='box', fillcolor='#D6E4FF', color='#2F54EB')
        c.node('b_dec', '解码器特征\n[B, C, H, W]\n(语义特征)', shape='box', fillcolor='#F5F5F5', color='#595959')
        c.node('b_cat', '通道拼接\n(Concat)', shape='box', style='filled', fillcolor='#FFE7BA', color='#D46B08')
        
        c.edge('b_enc', 'b_cat', label=' 直连')
        c.edge('b_dec', 'b_cat', label=' 语义传递')

    # =========================================================================
    # 子图 2：单流 GCA 机制
    # =========================================================================
    with dot.subgraph(name='cluster_single_stream') as c:
        c.attr(label='(b) 单流 GCA 结构', color='gray', bgcolor='#FAFAFA', style='dashed,rounded')
        
        c.node('s_enc', '编码器特征\n[B, C, H, W]', shape='box', fillcolor='#D6E4FF', color='#2F54EB')
        c.node('s_gca', 'GCA 注意力模块\n(计算曲率权重)', shape='box', fillcolor='#E6FFFB', color='#13C2C2')
        c.node('s_dec', '解码器特征\n[B, C, H, W]', shape='box', fillcolor='#F5F5F5', color='#595959')
        c.node('s_cat', '通道拼接\n(Concat)', shape='box', style='filled', fillcolor='#FFE7BA', color='#D46B08')
        
        c.edge('s_enc', 's_gca', label=' 输入')
        c.edge('s_gca', 's_cat', label=' 特征流')
        c.edge('s_dec', 's_cat', label=' 语义传递')

    # =========================================================================
    # 子图 3：双流 DSPG 机制
    # =========================================================================
    with dot.subgraph(name='cluster_dual_stream') as c:
        c.attr(label='(c) 本文双流 DSPG 架构 (Ours)', color='green', bgcolor='#F6FFED', style='filled,dashed,rounded')
        
        c.node('d_enc', '编码器特征\n[B, C, H, W]', shape='box', fillcolor='#D6E4FF', color='#2F54EB')
        c.node('d_gca', 'GCA 注意力模块\n(双流分流)', shape='box', fillcolor='#E6FFFB', color='#13C2C2')
        c.node('d_dec', '解码器特征\n[B, C, H, W]', shape='box', fillcolor='#F5F5F5', color='#595959')
        c.node('d_prod', '×', shape='circle', fillcolor='#FFF0F6', color='#EB2F96', width='0.4', height='0.4')
        c.node('d_cat', '通道拼接\n(Concat)', shape='box', style='filled', fillcolor='#FFE7BA', color='#D46B08')
        
        c.edge('d_enc', 'd_gca', label=' 输入')
        c.edge('d_gca', 'd_cat', label=' 【流一】特征增强流', color='#13C2C2', fontcolor='#13C2C2', weight='2')
        c.edge('d_gca', 'd_prod', label=' 【流二】物理先验流\n[B, 1, H, W]', color='#EB2F96', fontcolor='#EB2F96', style='bold')
        c.edge('d_dec', 'd_prod', label=' 语义特征')
        c.edge('d_prod', 'd_cat', label=' 先验引导特征', color='#73D13D', style='dashed')

    # 【切回矢量 PDF】完美解决乱码后，重新使用学术最高规范的 PDF 格式输出
    output_filename = 'architecture_compare_chinese_vector'
    dot.render(output_filename, format='pdf', cleanup=True)
    print(f"🚀 【终极成果斩获】高清无损、全中文的 PDF 矢量架构对比图已生成：{output_filename}.pdf")

if __name__ == '__main__':
    # 调用全新的矢量 PDF 函数
    draw_chinese_architecture_comparison_vector_pdf()