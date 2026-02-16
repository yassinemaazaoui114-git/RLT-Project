"""
PDF Report Generator for Breast Cancer Predictions
"""
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from datetime import datetime
from pathlib import Path
import io

class PDFReportGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#2563eb'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='CustomSubtitle',
            parent=self.styles['Normal'],
            fontSize=12,
            textColor=colors.HexColor('#64748b'),
            spaceAfter=20,
            alignment=TA_CENTER,
            fontName='Helvetica'
        ))
        
        # Section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#1e40af'),
            spaceAfter=12,
            spaceBefore=20,
            fontName='Helvetica-Bold'
        ))
        
        # Warning style
        self.styles.add(ParagraphStyle(
            name='Warning',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#dc2626'),
            spaceAfter=20,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))

    def generate_report(self, patient_data: dict, results: dict) -> bytes:
        """
        Generate a comprehensive PDF report.
        
        Args:
            patient_data: Dict with patient_id, date, and measurements
            results: Dict with model predictions (from /api/compare endpoint)
        
        Returns:
            bytes: PDF file content
        """
        # Create PDF in memory
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72,
        )
        
        # Container for PDF elements
        elements = []
        
        # Add header
        elements.extend(self._create_header())
        
        # Add warning banner
        elements.append(self._create_warning_banner())
        
        # Add patient information
        elements.extend(self._create_patient_info(patient_data))
        
        # Add consensus result
        elements.extend(self._create_consensus_section(results))
        
        # Add individual model results
        elements.extend(self._create_model_results_section(results))
        
        # Add measurements table
        elements.extend(self._create_measurements_section(patient_data))
        
        # Add footer
        elements.append(Spacer(1, 0.5*inch))
        elements.append(self._create_footer())
        
        # Build PDF
        doc.build(elements, onFirstPage=self._add_page_number, onLaterPages=self._add_page_number)
        
        # Get PDF bytes
        pdf_bytes = buffer.getvalue()
        buffer.close()
        
        return pdf_bytes
    
    def _create_header(self) -> list:
        """Create report header."""
        elements = []
        
        title = Paragraph("🩺 Breast Cancer AI Diagnostic Report", self.styles['CustomTitle'])
        elements.append(title)
        
        subtitle = Paragraph("Multi-Model Machine Learning Analysis", self.styles['CustomSubtitle'])
        elements.append(subtitle)
        
        elements.append(Spacer(1, 0.3*inch))
        
        return elements
    
    def _create_warning_banner(self) -> Paragraph:
        """Create warning banner."""
        warning_text = """
        <b>⚠️ MEDICAL PROFESSIONAL USE ONLY</b><br/>
        This AI-generated report is for diagnostic assistance only and must be reviewed by a qualified medical professional.
        """
        return Paragraph(warning_text, self.styles['Warning'])
    
    def _create_patient_info(self, patient_data: dict) -> list:
        """Create patient information section."""
        elements = []
        
        elements.append(Paragraph("Patient Information", self.styles['SectionHeader']))
        
        data = [
            ['Patient ID:', patient_data.get('patient_id', 'N/A')],
            ['Test Date:', patient_data.get('date', datetime.now().strftime('%Y-%m-%d'))],
            ['Report Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        ]
        
        table = Table(data, colWidths=[2*inch, 4*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f1f5f9')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#334155')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('PADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0'))
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 0.3*inch))
        
        return elements
    
    def _create_consensus_section(self, results: dict) -> list:
        """Create consensus result section."""
        elements = []
        
        elements.append(Paragraph("Diagnostic Consensus", self.styles['SectionHeader']))
        
        consensus = results.get('consensus', 'Unknown')
        comparison = results.get('comparison', {})
        
        agree_count = sum(1 for m in comparison.values() if m['diagnosis'] == consensus)
        total_count = len(comparison)
        
        consensus_color = colors.HexColor('#dc2626') if consensus == 'Malignant' else colors.HexColor('#059669')
        bg_color = colors.HexColor('#fee2e2') if consensus == 'Malignant' else colors.HexColor('#d1fae5')
        
        consensus_data = [
            [Paragraph(f"<b>🎯 CONSENSUS: {consensus.upper()}</b>", self.styles['Normal'])],
            [Paragraph(f"{agree_count}/{total_count} models agree", self.styles['Normal'])],
        ]
        
        consensus_table = Table(consensus_data, colWidths=[6*inch])
        consensus_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), bg_color),
            ('TEXTCOLOR', (0, 0), (-1, -1), consensus_color),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (0, 0), 16),
            ('FONTSIZE', (0, 1), (0, 1), 12),
            ('PADDING', (0, 0), (-1, -1), 15),
            ('BOX', (0, 0), (-1, -1), 2, consensus_color),
        ]))
        
        elements.append(consensus_table)
        elements.append(Spacer(1, 0.3*inch))
        
        return elements
    
    def _create_model_results_section(self, results: dict) -> list:
        """Create individual model results section."""
        elements = []
        
        elements.append(Paragraph("Individual Model Predictions", self.styles['SectionHeader']))
        
        comparison = results.get('comparison', {})
        
        table_data = [['Model', 'Diagnosis', 'Confidence', 'Malignant Probability']]
        
        for model_name, model_result in comparison.items():
            diagnosis = model_result['diagnosis']
            confidence = f"{model_result['confidence']:.1f}%"
            mal_prob = f"{model_result['malignant_probability']:.1f}%"
            
            table_data.append([model_name, diagnosis, confidence, mal_prob])
        
        table = Table(table_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 2*inch])
        
        style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2563eb')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8fafc')),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor('#334155')),
            ('ALIGN', (0, 1), (0, -1), 'LEFT'),
            ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('PADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')]),
        ])
        
        for i, (model_name, model_result) in enumerate(comparison.items(), start=1):
            if model_result['diagnosis'] == 'Malignant':
                style.add('TEXTCOLOR', (1, i), (1, i), colors.HexColor('#dc2626'))
                style.add('FONTNAME', (1, i), (1, i), 'Helvetica-Bold')
            else:
                style.add('TEXTCOLOR', (1, i), (1, i), colors.HexColor('#059669'))
                style.add('FONTNAME', (1, i), (1, i), 'Helvetica-Bold')
        
        table.setStyle(style)
        elements.append(table)
        elements.append(Spacer(1, 0.4*inch))
        
        return elements
    
    def _create_measurements_section(self, patient_data: dict) -> list:
        """Create tumor measurements section."""
        elements = []
        
        elements.append(Paragraph("Tumor Measurements", self.styles['SectionHeader']))
        
        measurements = patient_data.get('measurements', {})
        
        if not measurements:
            elements.append(Paragraph("No measurement data available.", self.styles['Normal']))
            return elements
        
        categories = {
            'Mean': [k for k in measurements.keys() if k.endswith('_mean')],
            'Standard Error': [k for k in measurements.keys() if k.endswith('_se')],
            'Worst': [k for k in measurements.keys() if k.endswith('_worst')],
        }
        
        for category, keys in categories.items():
            if not keys:
                continue
            
            elements.append(Paragraph(f"{category} Features", self.styles['Normal']))
            elements.append(Spacer(1, 0.1*inch))
            
            table_data = []
            for key in keys:
                display_name = key.replace('_', ' ').title()
                value = f"{measurements[key]:.4f}"
                table_data.append([display_name, value])
            
            mid = len(table_data) // 2
            combined_data = []
            for i in range(mid):
                row = table_data[i] + (table_data[i + mid] if i + mid < len(table_data) else ['', ''])
                combined_data.append(row)
            
            table = Table(combined_data, colWidths=[2*inch, 1*inch, 2*inch, 1*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f1f5f9')),
                ('BACKGROUND', (2, 0), (2, -1), colors.HexColor('#f1f5f9')),
                ('ALIGN', (0, 0), (0, -1), 'LEFT'),
                ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
                ('ALIGN', (2, 0), (2, -1), 'LEFT'),
                ('ALIGN', (3, 0), (3, -1), 'RIGHT'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('PADDING', (0, 0), (-1, -1), 4),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0'))
            ]))
            
            elements.append(table)
            elements.append(Spacer(1, 0.2*inch))
        
        return elements
    
    def _create_footer(self) -> Paragraph:
        """Create report footer."""
        footer_text = f"""
        <para align="center">
        <i>Generated by AI Diagnostic System on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}.<br/>
        For medical professional use only.</i>
        </para>
        """
        footer_style = ParagraphStyle(
            'Footer',
            parent=self.styles['Normal'],
            fontSize=8,
            textColor=colors.HexColor('#94a3b8'),
            alignment=TA_CENTER
        )
        return Paragraph(footer_text, footer_style)
    
    def _add_page_number(self, canvas, doc):
        """Add page numbers."""
        canvas.saveState()
        canvas.setFont('Helvetica', 9)
        page_num = canvas.getPageNumber()
        text = f"Page {page_num}"
        canvas.drawRightString(7.5*inch, 0.5*inch, text)
        canvas.restoreState()
