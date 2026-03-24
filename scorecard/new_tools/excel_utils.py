import os
import shutil
import pandas as pd
from openpyxl import load_workbook
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import io
from openpyxl.drawing.image import Image
from openpyxl.utils import get_column_letter
from openpyxl.utils.cell import coordinate_from_string, column_index_from_string
from openpyxl.styles import Alignment
from openpyxl.formatting.rule import ColorScaleRule
from PIL import Image as PILImage

def get_cell(row,col):
    """
        外包cell,针对外部excel写入时需要用到这个转换
    """
    return f'{get_column_letter(row)}{col}'



def insert_matplotlib_fig(worksheet, fig, position, scale=0.1):
    """
    将 matplotlib 图表插入到指定工作表的指定单元格位置。

    Parameters
    ----------
    worksheet : openpyxl.worksheet.worksheet.Worksheet
        目标工作表对象，即要插入图片的 sheet。
    fig : matplotlib.figure.Figure
        待插入的 matplotlib 图像对象。
    position : str
        插入位置的左上角单元格坐标，例如 'E1'、'B5'。
    scale : float, default 0.1
        显示缩放比例（0~1 之间较为常用），会同时按该比例缩放宽和高。

    Returns
    -------
    None
        函数直接在工作表中插入图片，不返回对象。
    """
    img_data = io.BytesIO()
    fig.savefig(img_data, format='png', bbox_inches='tight', dpi=300)
    img_data.seek(0)
    img = Image(img_data)
    img.width = int(img.width * scale)
    img.height = int(img.height * scale)
    worksheet.add_image(img, position)



def estimate_cell_dimensions(fig, scale=0.1, col_width_px=60, row_height_px=20):
    """
    估算某个 matplotlib 图像在指定缩放比例下，大致会占用多少行列的单元格。

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        待估算的图像对象。
    scale : float, default 0.1
        显示缩放比例，和 `insert_matplotlib_fig` 中的 scale 含义一致。
    col_width_px : int, default 60
        单个 Excel 列宽对应的大致像素值（根据你实际 Excel 设置调整），
        这里默认认为一列约 60 像素宽。
    row_height_px : int, default 20
        单个 Excel 行高对应的大致像素值，默认认为一行约 20 像素高。

    Returns
    -------
    tuple
        (num_rows, num_cols)，即建议合并的行数和列数，可直接用于 `merge_cells_from_start`。
    """
    # 获取图片原始像素尺寸
    img_data = io.BytesIO()
    fig.savefig(img_data, format='png', bbox_inches='tight', dpi=300) 
    img_data.seek(0)
    
    pil_img = PILImage.open(img_data)
    orig_width_px, orig_height_px = pil_img.size
    
    # 应用缩放
    display_width_px = int(orig_width_px * scale)
    display_height_px = int(orig_height_px * scale)
    
    # 计算行列数（向上取整）
    num_cols = max(1, display_width_px // col_width_px)
    num_rows = max(1, display_height_px // row_height_px)+1 # +1确保美观性
    
    # # 宽高比保护（避免图片被过度拉伸）
    # aspect_ratio = display_width_px / display_height_px
    # if num_cols / num_rows > aspect_ratio * 1.5:
    #     num_cols = int(num_rows * aspect_ratio)
    
    return num_rows, num_cols



def merge_cells_from_start(worksheet, start_cell, num_rows, num_cols, center=True):
    """
    从指定起始单元格开始，向右向下合并指定数量的行和列。

    Parameters
    ----------
    worksheet : openpyxl.worksheet.worksheet.Worksheet
        目标工作表对象。
    start_cell : str
        合并区域的左上角单元格，例如 'J1'。
    num_rows : int
        要合并的行数（正整数），包括起始行在内。
    num_cols : int
        要合并的列数（正整数），包括起始列在内。
    center : bool, default True
        是否将合并后左上角单元格的对齐方式设置为居中（水平/垂直）。

    Returns
    -------
    str
        合并后的单元格区域字符串，比如 'J1:L10'。

    """
    # 验证参数
    if num_rows <= 0 or num_cols <= 0:
        raise ValueError("num_rows 和 num_cols 必须为正整数")
    
    # 解析起始单元格
    try:
        start_col_letter, start_row = coordinate_from_string(start_cell)
        start_col_idx = column_index_from_string(start_col_letter)
    except Exception as e:
        raise ValueError(f"无效的起始单元格: {start_cell}") from e
    
    # 计算结束单元格位置
    end_row = start_row + num_rows - 1
    end_col_idx = start_col_idx + num_cols - 1
    
    # Excel 边界检查（2016+ 最大支持 1048576 行, 16384 列）
    if end_row > 1048576:
        raise ValueError(f"结束行号 {end_row} 超过 Excel 最大行数限制")
    if end_col_idx > 16384:
        raise ValueError(f"结束列号 {end_col_idx} 超过 Excel 最大列数限制")
    
    # 转换列索引为字母
    end_col_letter = get_column_letter(end_col_idx)
    
    # 构造合并范围，如 "J1:L10"
    merge_range = f"{start_cell}:{end_col_letter}{end_row}"
    
    # 执行合并
    worksheet.merge_cells(merge_range)
    
    # 设置居中对齐
    if center:
        worksheet[start_cell].alignment = Alignment(
            horizontal='center',
            vertical='center'
        )
    
    return merge_range

def apply_percentage_format(ws, start_cell, end_cell):
    """
    将指定单元格区域设置为“百分比格式，保留两位小数”。

    Parameters
    ----------
    ws : openpyxl.worksheet.worksheet.Worksheet
        目标工作表对象。
    start_cell : str
        区域左上角单元格地址，例如 'E2'。
    end_cell : str
        区域右下角单元格地址，例如 'Z50'。

    Returns
    -------
    None
        直接修改工作表中单元格的格式。
    """
    for row in ws[start_cell:end_cell]:
        for cell in row:
            cell.number_format = '0.00%'  # 百分比格式，保留两位小数

def apply_color_scale(ws, start_cell, end_cell, min_color='FFFFFF', max_color='F8696B'):
    """
    为指定单元格区域添加双色阶条件格式（颜色渐变）。

    Parameters
    ----------
    ws : openpyxl.worksheet.worksheet.Worksheet
        目标工作表对象。
    start_cell : str
        区域左上角单元格地址。
    end_cell : str
        区域右下角单元格地址。
    min_color : str, default 'FFFFFF'
        最小值对应的颜色（16 进制 RGB），默认白色。
    max_color : str, default 'F8696B'
        最大值对应的颜色（16 进制 RGB），默认红色。

    Returns
    -------
    None
        直接在工作表中添加条件格式规则。
    """
    rule = ColorScaleRule(
        start_type='min', start_color=min_color,
        end_type='max', end_color=max_color
    )
    ws.conditional_formatting.add(f'{start_cell}:{end_cell}', rule)

def excel_padding_format(left,up,right,down,df):
    """
    根据“左上右下”要排除的行列数，计算 DataFrame 在 Excel 中数据区域的起始/结束单元格。

    Parameters
    ----------
    left : int
        左侧需要排除的列数（例如排除一列“行号/索引”）。
    up : int
        上侧需要排除的行数（例如排除表头行）。
    right : int
        右侧需要排除的列数。
    down : int
        下侧需要排除的行数。
    df : pd.DataFrame
        对应的数据表，用于获取总行数和列数。

    Returns
    -------
    tuple
        (start_cell, end_cell)，例如 ('B2', 'E101')，可直接用于格式设置函数。
    """

    start_cell = f'{get_column_letter(left+1)}{up+1}'
    end_cell = f'{get_column_letter(df.shape[1] -right)}{df.shape[0] -down + 1}'
    return start_cell,end_cell


def copy_report_config_to_save_path(save_path, config_path="config/report_config.xlsx", save_name="report_config.xlsx"):
    """
    将模板报告（如 report_config.xlsx）从模板目录复制到目标保存目录。

    Parameters
    ----------
    save_path : str
        目标保存目录路径，如果不存在会自动创建。
    config_path : str, default "config/report_config.xlsx"
        源模板文件路径，可以是相对路径或绝对路径。
    save_name : str, default "report_config.xlsx"
        复制到目标目录后的文件名。

    Returns
    -------
    str
        复制后的目标文件完整路径。
    """
    # 确保SAVE_PATH目录存在
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # 构建目标文件路径
    target_file = os.path.join(save_path, save_name)
    
    # 检查目标文件是否已存在，如果存在则删除
    if os.path.exists(target_file):
        os.remove(target_file)
        print(f"已删除原有文件: {target_file}")
    
    # 复制文件
    shutil.copy2(config_path, target_file)
    
    print(f"已将 {config_path} 复制到 {target_file}")
    return target_file


def insert_dataframe_to_excel(
    excel_file_path,
    df: pd.DataFrame,
    start_row: int,
    start_col: int,
    sheet_name: str = None,
    include_index: bool = False,
    include_columns: bool = True,
    preserve_format: bool = True
):
    """
    将 DataFrame 数据写入到已有 Excel 文件的指定起始行列位置。

    Parameters
    ----------
    excel_file_path : str
        目标 Excel 文件路径（要求已存在，会在原文件基础上写入）。
    df : pd.DataFrame
        待写入的数据表。
    start_row : int
        起始行号（从 1 开始），对应 Excel 的实际行号。
    start_col : int
        起始列号（从 1 开始），对应 Excel 的实际列号。
    sheet_name : str or None, default None
        目标工作表名称：
        - 为 None：使用活动工作表；
        - 为指定名称：若存在则写入该表，不存在则新建同名表。
    include_index : bool, default False
        是否将 DataFrame 的索引一并写入到 Excel。
    include_columns : bool, default True
        是否写入列名作为表头。
    preserve_format : bool, default True
        是否保留目标单元格原有的格式（字体、颜色、对齐、边框、数字格式等）。
        - True：仅覆盖单元格值，尽量保留原格式；
        - False：直接覆盖值，不做格式恢复。

    Returns
    -------
    bool
        True 表示写入成功，False 表示写入过程中出现异常（并在控制台打印错误信息）。
    """
    try:
        # 预处理DataFrame，将复杂数据类型转换为字符串
        df_processed = df.copy()
        for col in df_processed.columns:
            if df_processed[col].dtype == 'object':
                # 检查是否包含Interval对象或其他复杂类型
                df_processed[col] = df_processed[col].astype(str)
        
        # 处理索引
        if include_index and df_processed.index.dtype == 'object':
            df_processed.index = df_processed.index.astype(str)
        
        workbook = load_workbook(excel_file_path)
        
        if sheet_name is None:
            worksheet = workbook.active
        else:
            if sheet_name in workbook.sheetnames:
                worksheet = workbook[sheet_name]
            else:
                worksheet = workbook.create_sheet(sheet_name)
        
        data_to_insert = []
        
        if include_columns:
            if include_index:
                columns = [df_processed.index.name if df_processed.index.name else ''] + list(df_processed.columns)
                data_to_insert.append(columns)
            else:
                data_to_insert.append(list(df_processed.columns))
        
        for idx, row in df_processed.iterrows():
            row_data = []
            if include_index:
                row_data.append(idx)
            row_data.extend(row.values)
            data_to_insert.append(row_data)
        

        # 将数据插入到指定位置
        for i, row_data in enumerate(data_to_insert):
            
            for j, value in enumerate(row_data):
                cell_row = start_row + i
                cell_col = start_col + j
                
                # 这里将能转换为数字的都转换为数字了，防止当一列中因为控制替换为了'/'导致出错
                if type(value) == str:
                    try:
                        value = float(value)
                    except:
                        pass

                # 获取目标单元格
                target_cell = worksheet.cell(row=cell_row, column=cell_col)
                
                # 如果保持格式，先保存当前单元格的格式
                if preserve_format:
                    # 保存原有格式（使用copy()方法）
                    original_font = target_cell.font.copy() if target_cell.font else None
                    original_fill = target_cell.fill.copy() if target_cell.fill else None
                    original_alignment = target_cell.alignment.copy() if target_cell.alignment else None
                    original_border = target_cell.border.copy() if target_cell.border else None
                    original_number_format = target_cell.number_format
                
                # 处理各种数据类型
                if pd.isna(value):
                    value = ""
                elif isinstance(value, (np.integer, np.floating)):
                    value = value.item()
                elif isinstance(value, str) and value == 'nan':
                    value = ""
                # 其他类型已经在预处理中转换为字符串
                
                # 更新单元格的值
                target_cell.value = value
                
                # 如果保持格式，恢复原有格式
                if preserve_format:
                    if original_font:
                        target_cell.font = original_font
                    if original_fill:
                        target_cell.fill = original_fill
                    if original_alignment:
                        target_cell.alignment = original_alignment
                    if original_border:
                        target_cell.border = original_border
                    if original_number_format:
                        target_cell.number_format = original_number_format
        
        workbook.save(excel_file_path)
        workbook.close()
        
        print(f"成功将DataFrame数据插入到 {excel_file_path} 的 {sheet_name or '活动工作表'} 工作表")
        print(f"插入位置: 行{start_row}, 列{start_col}")
        print(f"数据大小: {len(data_to_insert)}行 x {len(data_to_insert[0]) if data_to_insert else 0}列")
        
        return True
        
    except Exception as e:
        print(f"插入数据时发生错误: {str(e)}")
        return False


def insert_dataframe_to_excel_advanced(
    excel_file_path,
    df: pd.DataFrame,
    start_cell: str,
    sheet_name: str = None,
    include_index: bool = False,
    include_columns: bool = True,
    preserve_format: bool = True
):
    """
    基于“单元格坐标”（如 'A1'）来指定起始位置，将 DataFrame 写入到 Excel。

    本函数是对 `insert_dataframe_to_excel` 的封装：将 'A1' 形式的坐标转换为行列号后调用基础函数。

    Parameters
    ----------
    excel_file_path : str
        目标 Excel 文件路径。
    df : pd.DataFrame
        待写入的数据表。
    start_cell : str
        起始单元格坐标，例如 'A1'、'B5' 等。
    sheet_name : str or None, default None
        目标工作表名称，含义同 `insert_dataframe_to_excel`。
    include_index : bool, default False
        是否写入索引。
    include_columns : bool, default True
        是否写入列名。
    preserve_format : bool, default True
        是否保留原有单元格格式。

    Returns
    -------
    bool
        True 表示写入成功，False 表示失败。
    """
    try:
        # 将Excel单元格格式转换为行列号
        from openpyxl.utils import coordinate_to_tuple
        start_row, start_col = coordinate_to_tuple(start_cell)
        
        # 调用基础函数
        return insert_dataframe_to_excel(excel_file_path, df, start_row, start_col, 
                                       sheet_name, include_index, include_columns, preserve_format)
        
    except Exception as e:
        print(f"解析单元格位置时发生错误: {str(e)}")
        return False



if __name__ == "__main__":
    # example_usage()

    sample_df1 = pd.DataFrame({
        '特征名': ['feature1', 'feature2', 'feature3'],
        '重要性': [0.5, 0.3, 0.2],
        'IV值': [0.1, 0.05, 0.03]
    })
    sample_df2 = pd.DataFrame({
        '特征名': ['feature4', 'feature5', 'feature6'],
        '重要性': [0.5, 0.3, 0.2],
        'IV值': [0.1, 0.05, '/']
    })

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10,5))
    plt.bar(sample_df1['特征名'], sample_df1['重要性'])
    plt.xlabel('feature')
    plt.ylabel('iv')
    plt.title('feature iv')


    """
        以下包括了上述所有函数的示例用法，基于sample_df1,2作为示例数据,fig作为示例图片
    """
    # 这里是个性化写入相关的函数示例
    with pd.ExcelWriter(r'test_example.xlsx', engine='openpyxl') as writer:
        sample_df1.to_excel(writer, sheet_name='Sheet1', index=False)
        ws = writer.sheets['Sheet1']
        start_cell,end_cell = excel_padding_format(1,1,0,0,sample_df1) #选择数字区域，不要包含标题行和标题列，可以自定义范围，四个参数为左上右下
        apply_percentage_format(ws, start_cell, end_cell) # 修改单元格格式
        apply_color_scale(ws, start_cell, end_cell) # 修改单元格色阶
        insert_matplotlib_fig(ws, fig, 'E1') # 插入图片
        nrows,ncols = estimate_cell_dimensions(fig) # 估算图片占用的行列数
        merge_cells_from_start(ws, 'E1', nrows, ncols) # 合并单元格

    # 这里展示一个基于模板进行二次写入的示例，这个的试用场景就是联合建模的时候有要求联合建模的模型报告，需要基于模板进行二次写入

    # 设置路径
    import os
    save_path = os.getcwd() # 采用当前路径来做，因为个性化写入是写在了当前的路径下的
    excel_file = copy_report_config_to_save_path(save_path, config_path="test_example.xlsx",save_name = "test_example2.xlsx")
    
    # 插入数据到Excel,这里展示的是指定单元格的形式，指定单元格的写入形式是在在行列版本外面进行的套壳，行列版本为insert_dataframe_to_excel
    success = insert_dataframe_to_excel_advanced(
        excel_file, 
        sample_df2, 
        start_cell='A1',  # 从A1单元格开始
        sheet_name='Sheet1',
        include_index=False,
        include_columns=True
    )
    