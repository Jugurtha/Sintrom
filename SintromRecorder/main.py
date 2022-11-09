import sys
from PyQt5.uic import loadUi
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox, QDialog
import pandas as pd
import PandasModel as pm
import datetime


class AddRecordDialog(QDialog):
    def __init__(self) -> None:
        super(AddRecordDialog, self).__init__()
        loadUi('./AddRecordDialog.ui', self)
        self.date = self.findChild(QtWidgets.QDateEdit, "date")
        self.inr = self.findChild(QtWidgets.QDoubleSpinBox, "inr")
        self.tp = self.findChild(QtWidgets.QDoubleSpinBox, "tp")
        self.weekFrame = self.findChild(QtWidgets.QFrame, "weekFrame")
        self.dosage = self.findChild(QtWidgets.QComboBox, "dosage")
        self.dosage.activated.connect(lambda : self.weekFrame.setEnabled(True) if self.dosage.currentText()=='Other' else self.weekFrame.setEnabled(False))
        self.sunday = self.findChild(QtWidgets.QComboBox, "sunday")
        self.monday = self.findChild(QtWidgets.QComboBox, "monday")
        self.tuesday = self.findChild(QtWidgets.QComboBox, "tuesday")
        self.wednesday = self.findChild(QtWidgets.QComboBox, "wednesday")
        self.thursday = self.findChild(QtWidgets.QComboBox, "thursday")
        self.friday = self.findChild(QtWidgets.QComboBox, "friday")
        self.saturday = self.findChild(QtWidgets.QComboBox, "saturday")
        self.today = self.findChild(QtWidgets.QComboBox, "today")
        self.todayDIfferent = self.findChild(QtWidgets.QGroupBox, "todayDIfferent")
    
    def frac_to_float(self, text) -> float:
        if text == '1/2':
            return 0.5
        elif text == '1/4':
            return 0.25
        elif text == '3/4':
            return 0.75
        elif text == '0':
            return 0.0

    def get_record(self) -> dict:
        ret = {
                "Date": self.date.date().toString('yyyy-MM-dd'),
                "TP": self.tp.value(),
                "INR": self.inr.value()
        }
           
        if self.dosage.currentText()=='1/2 everyday':
            ret.update({
            "LastSaturday": .5,
            "Sunday": .5,
            "Monday": .5,
            "Tuesday": .5,
            "Wednesday": .5,
            "Thursday": .5,
            "Friday": .5,
            "Saturday": .5
            })
        elif self.dosage.currentText()=='1/4 everyday':
            ret.update({
            "LastSaturday": .25,
            "Sunday": .25,
            "Monday": .25,
            "Tuesday": .25,
            "Wednesday": .25,
            "Thursday": .25,
            "Friday": .25,
            "Saturday": .25
            })
        elif self.dosage.currentText()=='1/4 every other day': 
            week = ['LastSaturday','Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday']
            doses = [0.0]*len(week)
            vals = [.25, 0]
            j = 0
            index_day = week.index(datetime.date(self.date.date().year(), self.date.date().month(), self.date.date().day()).strftime("%A"))
            for i in range(len(week)):
                doses[(i + index_day) % len(week)] = vals[j]
                j = (j + 1) % len(vals)
            
            weekDict = {week[i] : doses[i] for i in range(len(week))}
            ret.update(weekDict) 
        elif self.dosage.currentText()=='One day 1/4, one day 1/2':
            week = ['LastSaturday','Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday']
            doses = [0.0]*len(week)
            vals = [.25, 0.5]
            j = 0
            index_day = week.index(datetime.date(self.date.date().year(), self.date.date().month(), self.date.date().day()).strftime("%A"))
            for i in range(len(week)):
                doses[(i + index_day) % len(week)] = vals[j]
                j = (j + 1) % len(vals)
            
            weekDict = {week[i] : doses[i] for i in range(len(week))}
            ret.update(weekDict)
        elif self.dosage.currentText()=='1/2 everyday, except Monday and thursday 1/4':
            ret.update({
            "LastSaturday": .5,
            "Sunday": .5,
            "Monday": .25,
            "Tuesday": .5,
            "Wednesday": .5,
            "Thursday": .25,
            "Friday": .5,
            "Saturday": .5
            })
        elif self.dosage.currentText()=='1/4 everyday, except Monday and thursday 0':
            ret.update({
            "LastSaturday": .25,
            "Sunday": .25,
            "Monday": 0.0,
            "Tuesday": .25,
            "Wednesday": .25,
            "Thursday": 0.0,
            "Friday": .25,
            "Saturday": .25
            })
        elif self.dosage.currentText()=='1/4 everyday, except Monday and thursday 1/2':
            ret.update({
            "LastSaturday": .25,
            "Sunday": .25,
            "Monday": 0.5,
            "Tuesday": .25,
            "Wednesday": .25,
            "Thursday": 0.5,
            "Friday": .25,
            "Saturday": .25
            })
        elif self.dosage.currentText()=='1/2 everyday, except Monday and thursday 3/4':
            ret.update({
            "LastSaturday": .5,
            "Sunday": .5,
            "Monday": 0.75,
            "Tuesday": .5,
            "Wednesday": .5,
            "Thursday": 0.75,
            "Friday": .5,
            "Saturday": .5
            })
        elif self.dosage.currentText()=='Three days 1/4, Two days 0':# TODO Fix this
            week = ['LastSaturday','Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday']
            doses = [0.0]*len(week)
            vals = [.25,.25,.25, 0, 0]
            j = 0
            index_day = week.index(datetime.date(self.date.date().year(), self.date.date().month(), self.date.date().day()).strftime("%A"))
            for i in range(len(week)):
                doses[(i + index_day) % len(week)] = vals[j]
                j = (j + 1) % len(vals)
            
            weekDict = {week[i] : doses[i] for i in range(len(week))}
            ret.update(weekDict)
        elif self.dosage.currentText()=='Two days 1/2, one day 1/4':# TODO Fix this
            week = ['LastSaturday','Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday']
            doses = [0.0]*len(week)
            vals = [.5,.5,.25]
            j = 0
            index_day = week.index(datetime.date(self.date.date().year(), self.date.date().month(), self.date.date().day()).strftime("%A"))
            for i in range(len(week)):
                doses[(i + index_day) % len(week)] = vals[j]
                j = (j + 1) % len(vals)
            
            weekDict = {week[i] : doses[i] for i in range(len(week))}
            ret.update(weekDict)
        else:
            ret.update({
                "LastSaturday": self.frac_to_float(self.saturday.currentText()),
                "Sunday": self.frac_to_float(self.sunday.currentText()),
                "Monday": self.frac_to_float(self.monday.currentText()),
                "Tuesday": self.frac_to_float(self.tuesday.currentText()),
                "Wednesday": self.frac_to_float(self.wednesday.currentText()),
                "Thursday": self.frac_to_float(self.thursday.currentText()),
                "Friday": self.frac_to_float(self.friday.currentText()),
                "Saturday": self.frac_to_float(self.saturday.currentText())
            })
        
        ret.update({
            "Weekday": datetime.date(self.date.date().year(), self.date.date().month(), self.date.date().day()).strftime("%A")
        })
        ret.update({
            "DiffrentToday": self.todayDIfferent.isChecked(),
            "DosageToday" : self.frac_to_float(self.today.currentText()) if self.todayDIfferent.isChecked() else ret[ret["Weekday"]]
        })
        categories = ["1/2 everyday",# Done
                        "1/4 everyday",# Done
                        "1/4 every other day",# Done
                        "One day 1/4, one day 1/2",# Done
                        "1/2 everyday, except Monday and thursday 1/4",# Done
                        "1/4 everyday, except Monday and thursday 1/2",# Done
                        "1/4 everyday, except Monday and thursday 0",# Done
                        "1/2 everyday, except Monday and thursday 3/4",# Done
                        "Three days 1/4, Two days 0",#Sorte of done
                        "Two days 1/2, one day 1/4",#Sorte of done
                        "Other"]
        ret.update({"Category" : categories.index(self.dosage.currentText())})
        #print(ret)
        return ret



class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super(MainWindow, self).__init__()
        loadUi('./SintromRecorder.ui', self)

        self.file_loaded = False
        self.file_saved = True
        self.file_path = ''

        self.actionLoad_file = self.findChild(QtWidgets.QAction, "actionLoad_file")
        self.actionLoad_file.triggered.connect(self.load_file)

        self.actionSave_file = self.findChild(QtWidgets.QAction, "actionSave_file")
        self.actionSave_file.triggered.connect(self.save_file)

        self.actionExit = self.findChild(QtWidgets.QAction, "actionExit")
        self.actionExit.triggered.connect(self.safe_exit)
        
        
        self.actionExit = self.findChild(QtWidgets.QAction, "actionExit")
        self.actionExit.triggered.connect(self.safe_exit)
        
        self.actionAbout_Qt = self.findChild(QtWidgets.QAction, "actionAbout_Qt")
        self.actionAbout_Qt.triggered.connect(self.about_Qt)
        
        self.actionAbout_Sintrom_Recorder = self.findChild(QtWidgets.QAction, "actionAbout_Sintrom_Recorder")
        self.actionAbout_Sintrom_Recorder.triggered.connect(self.about_Sintrom_Recorder)
        
        self.tableView = self.findChild(QtWidgets.QTableView, "tableView")
        self.tableView.clicked.connect(lambda: self.actionDelete_record.setEnabled(True))

        
        self.actionAdd_new_record = self.findChild(QtWidgets.QAction, "actionAdd_new_record")
        self.actionAdd_new_record.triggered.connect(self.add_new_record)
        
        self.actionDelete_record = self.findChild(QtWidgets.QAction, "actionDelete_record")
        self.actionDelete_record.triggered.connect(self.delete_record)


    def load_file(self)-> None:
        if self.file_loaded and not self.file_saved:
            response = QMessageBox.question(self, "File Not saved", "Do you want to save changes to " + self.file_path)
            if response == QMessageBox.Yes:
                self.save_file()
            self.save_file()


        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, ok = QFileDialog.getOpenFileName(self,"Open CSV File", "","CSV Files (*.csv)", options=options)
        if ok:
            # print(fileName)
            self.file_path = fileName
            # print(self.data_df.info())
            # print(self.data_df.head())

            # Populate tableview
            self.model = pm.DataFrameModel(pd.read_csv(self.file_path, header=0, index_col=0))
            self.tableView.setModel(self.model)

            self.file_loaded = True
            self.actionSave_file.setEnabled(True)
            self.tableView.setEnabled(True)
            self.actionAdd_new_record.setEnabled(True)

            
    

    def save_file(self) -> None:
        self.model.dataFrame.to_csv(self.file_path)
        self.file_saved = True


    def safe_exit(self):
        if self.file_loaded and not self.file_saved:
            response = QMessageBox.question(self, "File Not saved", "Do you want to save changes to " + self.file_path )
            if response == QMessageBox.Yes:
                self.save_file()
        self.close()

    def closeEvent(self, event):
        if self.file_loaded and not self.file_saved:
            response = QMessageBox.question(self, "File Not saved", "Do you want to save changes to " + self.file_path )
            if response == QMessageBox.Yes:
                self.save_file()
        event.accept()

    def add_new_record(self):
        ard = AddRecordDialog()
        if ard.exec():
            self.model.addRow(ard.get_record())
            self.file_saved = False
        
    
    def delete_record(self):
        index = self.tableView.selectionModel().selectedRows()[0]  
        response = QMessageBox.question(self, "Delete record", "Are you sure you want to delete the selected record ?" )
        if response == QMessageBox.Yes:
            # print('Row %d is selected' % index.row())
            self.model.deleteRow(index)
            self.file_saved = False
    
    
    def about_Qt(self):
        QMessageBox.about(self, "About Qt", "This sofware was created using Qt5 and PyQt5.")

    
    def about_Sintrom_Recorder(self):
        QMessageBox.about(self, "About Sintrom Recorder", "This sofware is aimed to be used for recording Sintrom dosage and TP/INR of a Patient.")








if __name__ == '__main__':
    # print('Hello World !!!')
    app = QApplication(sys.argv)
    MainWindow = MainWindow()
    MainWindow.show()

    try:
        sys.exit(app.exec_())
    except:
        print("Exiting !!!")