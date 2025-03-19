/********************************************************************************
** Form generated from reading UI file 'deeplayout.ui'
**
** Created by: Qt User Interface Compiler version 6.8.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_DEEPLAYOUT_H
#define UI_DEEPLAYOUT_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_DeeplayoutClass
{
public:
    QMenuBar *menuBar;
    QToolBar *mainToolBar;
    QWidget *centralWidget;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *DeeplayoutClass)
    {
        if (DeeplayoutClass->objectName().isEmpty())
            DeeplayoutClass->setObjectName("DeeplayoutClass");
        DeeplayoutClass->resize(600, 400);
        menuBar = new QMenuBar(DeeplayoutClass);
        menuBar->setObjectName("menuBar");
        DeeplayoutClass->setMenuBar(menuBar);
        mainToolBar = new QToolBar(DeeplayoutClass);
        mainToolBar->setObjectName("mainToolBar");
        DeeplayoutClass->addToolBar(mainToolBar);
        centralWidget = new QWidget(DeeplayoutClass);
        centralWidget->setObjectName("centralWidget");
        DeeplayoutClass->setCentralWidget(centralWidget);
        statusBar = new QStatusBar(DeeplayoutClass);
        statusBar->setObjectName("statusBar");
        DeeplayoutClass->setStatusBar(statusBar);

        retranslateUi(DeeplayoutClass);

        QMetaObject::connectSlotsByName(DeeplayoutClass);
    } // setupUi

    void retranslateUi(QMainWindow *DeeplayoutClass)
    {
        DeeplayoutClass->setWindowTitle(QCoreApplication::translate("DeeplayoutClass", "Deeplayout", nullptr));
    } // retranslateUi

};

namespace Ui {
    class DeeplayoutClass: public Ui_DeeplayoutClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_DEEPLAYOUT_H
